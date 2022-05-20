#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from codes.datasets.dataset import DataProvider
from codes.search.searcher import Searcher
from codes.supernet.network import Supernet, FixedSupernet
from codes.utils.evaluate import get_batch_softmax_loss
from codes.utils.utils import get_cand_flops, get_pooling_flops

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))


class EvolutionSearcher(Searcher):

    def __init__(self, train_dataloader, val_dataloader, num_users, num_items, args):
        super(EvolutionSearcher, self).__init__(name='AUTOTOWER',
                                                train_dataloader=train_dataloader,
                                                val_dataloader=val_dataloader,
                                                max_epochs=args.search_epochs,
                                                num_users=num_users,
                                                num_items=num_items,
                                                reg=args.reg,
                                                args=args)

        self._train_dataprovider = DataProvider(self._train_dataloader)
        self._val_dataprovider = DataProvider(self._val_dataloader)

        self._select_num = args.select_num
        self._population_num = args.population_num
        self._m_prob = args.m_prob
        self._crossover_num = args.crossover_num
        self._mutation_num = args.mutation_num
        # self.flops_limit = args.flops_limit

        self._max_train_iters = args.max_train_iters
        self._max_test_iters = args.max_test_iters

        self._supernet = self._get_supernet(load_supernet=True)

        self._memory = []
        self._vis_dict = {}
        self._topk = self._select_num
        self._keep_topk = []
        self._candidates = []

        self._num_eval = 0

        self._countinous_illegal_time = 0

    def get_tag(self):
        tag = f'{self.name}' \
              f'_{self._select_num}_{self._population_num}_{self._crossover_num}_{self._mutation_num}_{self._m_prob}' \
              f'_{self._max_train_iters}_{self._max_test_iters}_{self._args.supernet_tag}'

        return tag

    @staticmethod
    def new_retrain_model(args, dataset, checkpoint_path, topk=1):
        checkpoint = torch.load(checkpoint_path)
        infos = checkpoint['vis_dict']
        archs = sorted([(arch, infos[arch]['loss']) for arch in infos if 'loss' in infos[arch]],
                       key=lambda x: infos[x[0]]['loss'])[:topk]

        return ((FixedSupernet(dataset,
                               user_tower_config=checkpoint['user_tower_config'],
                               item_tower_config=checkpoint['item_tower_config'],
                               pooling_config=checkpoint['pooling_config'],
                               arch=arch),
                 loss) for arch, loss in archs)

    def _save_checkpoint(self):
        if not os.path.exists(os.path.dirname(self.checkpoint_file_path)):
            os.makedirs(os.path.dirname(self.checkpoint_file_path))
        info = {}
        info['memory'] = self._memory
        info['candidates'] = self._candidates
        info['vis_dict'] = self._vis_dict
        info['keep_topk'] = self._keep_topk
        info['epoch'] = self._epoch
        info['search_time'] = self._search_time

        info['num_users'] = self._supernet.num_users
        info['num_items'] = self._supernet.num_items

        info['user_tower_config'] = self._supernet.user_tower_config
        info['item_tower_config'] = self._supernet.item_tower_config
        info['pooling_config'] = self._supernet.pooling_config

        info['reg'] = self._supernet.reg

        torch.save(info, self.checkpoint_file_path)
        self._logger.info(f'save checkpoint to {self.checkpoint_file_path}')

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file_path):
            return False
        info = torch.load(self.checkpoint_file_path)
        self._memory = info['memory']
        self._candidates = info['candidates']
        self._vis_dict = info['vis_dict']
        self._keep_topk = info['keep_topk']
        self._epoch = info['epoch'] + 1
        self._search_time = info['search_time']
        self._logger.info(f'load checkpoint from {self.checkpoint_file_path}')
        return True

    def _get_cand_loss(self, cand):
        # clear bn statics
        for m in self._supernet.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

        # train bn with training set (BN sanitize)
        self._supernet.train()
        for _ in tqdm(range(self._max_train_iters)):
            users, histories, items, labels = self._train_dataprovider.next()
            if self._args.use_gpu:
                users, histories, items, labels = users.cuda(), histories.cuda(), items.cuda(), labels.cuda()

            self._supernet(users, histories, items, cand)

            del users, histories, items, labels

        # inference on val dataset
        self._supernet.eval()
        # loss = 0
        # num_sample = 0
        losses = []
        with torch.no_grad():
            for step in tqdm(range(self._max_test_iters)):
                users, histories, items, labels = self._val_dataprovider.next()
                if self._args.use_gpu:
                    users, histories, items, labels = users.cuda(), histories.cuda(), items.cuda(), labels.cuda()
                # num_sample += labels.shape[0]

                preds, regs = self._supernet(users, histories, items, cand)
                _, loss = get_batch_softmax_loss(preds, labels)
                losses.append(loss)

            del users, histories, items, labels, preds, regs
        val_loss = torch.mean(torch.tensor(losses))
        # loss /= num_sample
        # loss = loss ** 0.5
        self._num_eval += 1
        if self._args.show_tfboard:
            self._writter.add_scalar('AUTOTOWER_search/val_loss', val_loss, self._num_eval)

        return val_loss.cpu().detach().item()

    def _is_legal(self, arch):
        assert self._countinous_illegal_time < 200, 'Continous get illegel architectures except 200 times. Skip search.'
        user_tower_arch, item_tower_arch, pooling_type = arch
        assert isinstance(user_tower_arch, tuple) and len(user_tower_arch) == self._supernet.user_tower_config.num_layers
        assert isinstance(item_tower_arch, tuple) and len(item_tower_arch) == self._supernet.item_tower_config.num_layers
        if arch not in self._vis_dict:
            self._vis_dict[arch] = {}
        info = self._vis_dict[arch]
        if 'visited' in info:
            return False

        self._countinous_illegal_time += 1

        if 'pooling_flops' not in info:
            info['pooling_flops'], info['pooling_params'] = get_pooling_flops(self._supernet.pooling_config.pooling_types[pooling_type],
                                                                              self._supernet._item_embedding_dim,
                                                                              self._supernet._seq_fields, self._supernet._seq_len)

        if 'user_tower_flops' not in info:
            info['user_tower_flops'], info['user_tower_params'] = get_cand_flops(user_tower_arch,
                                                                                 self._supernet.user_tower_config.blocks_types,
                                                                                 sum(self._supernet._user_fea_emb_dims) + \
                                                                                     self._supernet._seq_fields * \
                                                                                         self._supernet._item_embedding_dim)
            info['user_tower_flops'] += info['pooling_flops']
            info['user_tower_params'] += info['pooling_params']
        if 'item_tower_flops' not in info:
            info['item_tower_flops'], info['item_tower_params'] = get_cand_flops(item_tower_arch,
                                                                                 self._supernet.item_tower_config.blocks_types,
                                                                                 sum(self._supernet._item_fea_emb_dims))


        self._logger.info(f"user_tower {user_tower_arch} {info['user_tower_flops']} {info['user_tower_params']}")
        self._logger.info(f"item_tower {item_tower_arch} {info['item_tower_flops']} {info['item_tower_params']}")
        self._logger.info(f"pooling_type {pooling_type} {info['pooling_flops']} {info['pooling_params']}")

        if self._args.use_flops_limits:
            if self._args.use_total_flops_limits:
                total_flops = info['user_tower_flops'] + info['item_tower_flops']
                if total_flops > self._args.total_flops_limit:
                    self._logger.info(f'total flops limit exceed, {total_flops} > {self._args.total_flops_limit}')
                    return False
            else:
                if info['user_tower_flops'] > self._args.user_tower_flops_limit:
                    self._logger.info(f"user tower flops limit exceed, {info['user_tower_flops']} > {self._args.user_tower_flops_limit}")
                    return False
                if info['item_tower_flops'] > self._args.item_tower_flops_limit:
                    self._logger.info(f"item tower flops limit exceed, {info['item_tower_flops']} > {self._args.item_tower_flops_limit}")
                    return False

        if self._args.use_params_limits:
            if self._args.use_total_params_limits:
                total_params = info['user_tower_params'] + info['item_tower_params']
                if total_params > self._args.total_params_limit:
                    self._logger.info(f'total params limit exceed, {total_params} > {self._args.total_params_limit}')
                    return False
            else:
                if info['user_tower_params'] > self._args.user_tower_params_limit:
                    self._logger.info(f"user tower params limit exceed, {info['user_tower_params']} > {self._args.user_tower_params_limit}")
                    return False
                if info['item_tower_params'] > self._args.item_tower_params_limit:
                    self._logger.info(f"item tower params limit exceed, {info['item_tower_params']} > {self._args.item_tower_params_limit}")
                    return False

        info['loss'] = self._get_cand_loss(arch)

        info['visited'] = True

        self._countinous_illegal_time = 0
        return True

    def _update_topk(self, candidates, key, reverse=False):
        self._logger.info('select ......')
        self._keep_topk += candidates
        self._keep_topk.sort(key=key, reverse=reverse)
        self._keep_topk = self._keep_topk[:self._topk]

    def _stack_random_cand(self, random_func, *, batch_size=10):
        while True:
            cands = [random_func() for _ in range(batch_size)]
            for cand in cands:
                if cand not in self._vis_dict:
                    self._vis_dict[cand] = {}
            for cand in cands:
                yield cand

    def _get_random(self, num):
        self._logger.info('random select...')
        cand_iter = self._stack_random_cand(
            lambda: (tuple(np.random.randint(len(self._supernet.user_tower_config.blocks_types[i]))
                           for i in range(self._supernet.user_tower_config.num_layers)),
                     tuple(np.random.randint(len(self._supernet.item_tower_config.blocks_types[i]))
                           for i in range(self._supernet.item_tower_config.num_layers)),
                     np.random.randint(len(self._supernet.pooling_config.pooling_types))
                     )
        )
        while len(self._candidates) < num:
            cand = next(cand_iter)
            if not self._is_legal(cand):
                continue
            self._candidates.append(cand)
            self._logger.info(f'random {len(self._candidates)}/{num}')
        self._logger.info(f'random_num={len(self._candidates)}')

    def _get_mutation(self, mutation_num, m_prob):
        self._logger.info('mutation...')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            user_tower_cand, item_tower_cand, pooling_cand = choice(self._keep_topk)
            user_tower_cand, item_tower_cand = list(user_tower_cand), list(item_tower_cand)
            for i in range(self._supernet.user_tower_config.num_layers):
                if np.random.random_sample() < m_prob:
                    user_tower_cand[i] = np.random.randint(len(self._supernet.user_tower_config.blocks_types[i]))
            for i in range(self._supernet.item_tower_config.num_layers):
                if np.random.random_sample() < m_prob:
                    item_tower_cand[i] = np.random.randint(len(self._supernet.item_tower_config.blocks_types[i]))
            if np.random.random_sample() < m_prob:
                pooling_cand = np.random.randint(len(self._supernet.pooling_config.pooling_types))

            cand = (tuple(user_tower_cand), tuple(item_tower_cand), pooling_cand)
            return cand

        cand_iter = self._stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self._is_legal(cand):
                continue
            res.append(cand)
            self._logger.info(f'mutation {len(res)}/{mutation_num}')

        self._logger.info(f'mutation_num={len(res)}')
        return res

    def _get_crossover(self, crossover_num):
        self._logger.info('crossover...')
        res = []
        max_iters = 10 * crossover_num

        def random_func():
            u1, i1, p1 = choice(self._keep_topk)
            u2, i2, p2 = choice(self._keep_topk)
            return (tuple(choice([i, j]) for i, j in zip(u1, u2)),
                    tuple(choice([i, j]) for i, j in zip(i1, i2)),
                    choice([p1, p2]))

        cand_iter = self._stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self._is_legal(cand):
                continue
            res.append(cand)
            self._logger.info(f'crossover {len(res)}/{crossover_num}')

        self._logger.info(f'crossover_num={len(res)}')
        return res

    def search(self):
        self._logger.info(f'population_num={self._population_num}, '
                          f'select_num={self._select_num}, '
                          f'mutation_num={self._mutation_num}, '
                          f'crossover_num={self._crossover_num}, '
                          f'random_num={self._population_num - self._mutation_num - self._crossover_num}, '
                          f'max_epochs={self._max_epochs}')

        self._load_checkpoint()

        self._get_random(self._population_num)

        while self._epoch < self._max_epochs:
            t = time.time()
            self._memory.append([])
            for cand in self._candidates:
                self._memory[-1].append(cand)

            self._update_topk(self._candidates, key=lambda x: self._vis_dict[x]['loss'])

            self._logger.info(f'epoch={self._epoch}, top {len(self._keep_topk)} result')
            for i, cand in enumerate(self._keep_topk):
                self._logger.info(f"No.{i + 1} {cand} Top-1 loss={self._vis_dict[cand]['loss']}")
            mutation = self._get_mutation(self._mutation_num, self._m_prob)
            crossover = self._get_crossover(self._crossover_num)
            self._candidates = mutation + crossover
            self._get_random(self._population_num)

            cur_search_time = time.time() - t
            self._search_time += cur_search_time
            self._logger.info(f"epoch={self._epoch}, val_loss={self._vis_dict[self._keep_topk[0]]['loss']}, use_time={cur_search_time}")
            self._save_checkpoint()
            self._epoch += 1
