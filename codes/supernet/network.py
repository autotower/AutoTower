#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from codes.blocks.block import BLOCKS_LIB
from codes.blocks.pooling import POOLING_LIB
from codes.datasets.dataset import check_multihot_column
from codes.utils.utils import encode_block_types, CONFIG_PATH, SUPERNET_TAG, initialize_embeddings, \
    get_embeddings, encode_pooling_types
from codes.utils.evaluate import get_batch_softmax_loss, get_corrected_batch_softmax_loss

FakeDataset = collections.namedtuple('FakeDataset',
                                       ['num_users', 'num_items', 'name', 'user_fea_fields', 'item_fea_fields'])


class PoolingConfig:
    def __init__(self, config):
        self.pooling_types = config['types']
        self.embedding_dim = config['embedding_dim']
        self.seq_fields = config['seq_fields']
        self.seq_len = config['seq_len']
    
    @property
    def tag(self):
        return f'pooling_{self.embedding_dim}_{self.seq_fields}_{self.seq_len}_'  \
            f'{encode_pooling_types(self.pooling_types)}'


class TowerConfig:
    def __init__(self, tower_name, config):
        self.tower_name = tower_name

        self.num_layers = config['num_layers']
        self._all_layer_blocks = config['all_layer_blocks']
        self._first_layer_blocks = config['first_layer_blocks']
        self.blocks_types = [self._all_layer_blocks[:] for _ in range(1, self.num_layers)]
        self.blocks_types.insert(0, self._first_layer_blocks[:])
        self.embedding_dim = config['embedding_dim']
        self.block_in_dim = config['block_in_dim']
        self.tower_out_dim = config['tower_out_dim']
        self.reg = config['reg']


    @property
    def tag(self):
        return f"{self.tower_name}_{self.num_layers}_{self.embedding_dim}_{self.block_in_dim}_{self.tower_out_dim}_{self.reg}" \
               f"_{encode_block_types(self._first_layer_blocks)}_{encode_block_types(self._all_layer_blocks)}"


class Base(nn.Module):
    def __init__(self,
                 dataset,
                 user_tower_config,
                 item_tower_config,
                 pooling_config):
        super(Base, self).__init__()
        self._dataset_name = dataset.name
        self._num_users = dataset.num_users
        self._num_items = dataset.num_items
        if not isinstance(user_tower_config, dict):
            self._user_tower_config = user_tower_config
        else:
            self._user_tower_config = TowerConfig('user_tower', user_tower_config)
            
        if not isinstance(pooling_config, dict):
            self._pooling_config = pooling_config
        else:
            self._pooling_config = PoolingConfig(pooling_config)
            
        if not isinstance(item_tower_config, dict):
            self._item_tower_config = item_tower_config
        else:
            self._item_tower_config = TowerConfig('item_tower', item_tower_config)
        self._loss_type = 'batch_softmax'

        self._reg = self._user_tower_config.reg

        self._user_embedding_dim = self._user_tower_config.embedding_dim
        self._item_embedding_dim = self._item_tower_config.embedding_dim
        
        self._seq_fields = self._pooling_config.seq_fields
        self._seq_len = self._pooling_config.seq_len

        self._user_fea_fields = dataset.user_fea_fields
        # self._user_fea_emb_dims = dataset.user_fea_emb_dim
        # self._user_fea_emb_dims[0] = self._user_embedding_dim
        self._user_fea_emb_dims = [self._user_embedding_dim] * len(dataset.user_fea_fields)
        self._item_fea_fields = dataset.item_fea_fields
        # self._item_fea_emb_dims = dataset.item_fea_emb_dim
        # self._item_fea_emb_dims[0] = self._item_embedding_dim
        self._item_fea_emb_dims = [self._item_embedding_dim] * len(dataset.item_fea_fields)
        self._users_embedding = initialize_embeddings(fea_dims=self._user_fea_fields, emb_dims=self._user_fea_emb_dims,
                                                      multihot_fea=check_multihot_column(self._dataset_name, 'user'))
        self._items_embedding = initialize_embeddings(fea_dims=self._item_fea_fields, emb_dims=self._item_fea_emb_dims,
                                                      multihot_fea=check_multihot_column(self._dataset_name, 'item'))

        self._users_trans = nn.Sequential(nn.Linear(sum(self._user_fea_emb_dims) + self._seq_fields * self._item_embedding_dim,
                                                    self._user_tower_config.block_in_dim))
        self._items_trans = nn.Sequential(nn.Linear(sum(self._item_fea_emb_dims), self._item_tower_config.block_in_dim))

        # self._score_fc = nn.Linear(self._user_tower_config.tower_out_dim + self._item_tower_config.tower_out_dim, 1)
        
        # used to maintain appeared items for frequency estimation
        self._A = {}
        self._B = {}
        
        
    def save_correction_dict(self, path):
        """Save correction dict to file.
        
        Args:
            path: dict save file path
        """
        with open(path, 'wb') as f:
            pickle.dump(self._B, f)
            
    def load_correction_dict(self, B_path):
        """Load correction dict.
        
        Args:
            B_path: dict B path
        """
        with open(B_path, 'rb') as f:
            self._B = pickle.load(f)
        
    def compute_loss_corrected(self, inferences, labels, regs, items, t, alpha=0.01):
        labels = torch.reshape(labels, [-1, 1])
        preds, loss = get_corrected_batch_softmax_loss(inferences, labels, self._A, self._B, t,
                                                       items, self._num_items // labels.shape[0], alpha)
        return loss + regs

    def compute_loss(self, inferences, labels, regs):
        labels = torch.reshape(labels, [-1, 1])
        preds, loss = get_batch_softmax_loss(inferences, labels)

        return loss + regs

    def get_tag(self):
        return f'{self._user_tower_config.tag}_{self._item_tower_config.tag}_{self._pooling_config.tag}'

    def get_kwargs(self):
        dataset = FakeDataset(num_users=self._num_users, num_items=self._num_items, name=self._dataset_name,
                              user_fea_fields=self._user_fea_fields, item_fea_fields=self._item_fea_fields)
        kwargs = {
            'dataset': dataset,
            'user_tower_config': self.user_tower_config,
            'item_tower_config': self.item_tower_config,
            'pooling_config': self.pooling_config
        }
        return kwargs
    
    def pooling_history(self, histories):
        """
        :params histories: [B, nfields, L]
        """
        mask = (histories[:, 0, :] != 0)
        nfields = histories.shape[1]
        histories_embs = []
        for i in range(nfields):
            history = histories[:, i, :]
            histories_emb = self._items_embedding[i](history)  # [B, L, E]
            histories_embs.append(histories_emb)
            
        histories_embs = torch.cat(histories_embs, dim=-1)  # [B, L, field * E]
        return histories_embs, mask

    @property
    def num_items(self):
        return self._num_items

    @property
    def num_users(self):
        return self._num_users

    @property
    def reg(self):
        return self._reg

    @property
    def user_tower_config(self):
        return self._user_tower_config
    
    @property
    def pooling_config(self):
        return self._pooling_config

    @property
    def item_tower_config(self):
        return self._item_tower_config

    @property
    def train_use_time(self):
        return self._train_use_time

    @property
    def loss_type(self):
        return self._loss_type

    @train_use_time.setter
    def train_use_time(self, x):
        self._train_use_time = x


class Supernet(Base):
    def __init__(self,
                 dataset,
                 user_tower_config,
                 item_tower_config,
                 pooling_config,
                 ):
        super(Supernet, self).__init__(dataset=dataset,
                                       user_tower_config=user_tower_config,
                                       item_tower_config=item_tower_config,
                                       pooling_config=pooling_config)
        self._user_tower = self._build_tower(self._user_tower_config)
        self._item_tower = self._build_tower(self._item_tower_config)
        self._pooling_layer = self._build_pooling(self._pooling_config)
        
        self._train_use_time = 0

    def get_tag(self):
        tag = super(Supernet, self).get_tag()

        return f'{SUPERNET_TAG}_{tag}'

    def save_checkpoint(self, path, iters):
        data = {'state_dict': self.state_dict(), 'kwargs': self.get_kwargs(),'train_use_time': self._train_use_time,
                'correction_dict_A': self._A, 'correction_dict_B': self._B}
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, f"{self.get_tag()}#{iters:06}.pth.tar")
        torch.save(data, filename)
        # latest_file_name = os.path.join(path, f"{tag}checkpoint-latest.pth.tar")
        # torch.save(data, latest_file_name)

    def _build_tower(self, tower_config):
        layers = nn.ModuleList()
        num_inputs = 1
        for i in range(tower_config.num_layers):
            params = {
                'block_in_dim': tower_config.block_in_dim,
                'block_out_dim': tower_config.block_in_dim,
                'tower_out_dim': tower_config.tower_out_dim,
                'embedding_dim': tower_config.embedding_dim,
                'num_inputs': num_inputs,
                'is_first_layer': False,
                'is_last_layer': False
            }
            layer = nn.ModuleList()
            if i == 0:
                params['is_first_layer'] = True
            elif i == tower_config.num_layers - 1:
                params['is_last_layer'] = True
            for block in tower_config.blocks_types[i]:
                layer.append(BLOCKS_LIB[block](params))
            num_inputs += 1
            layers.append(layer)

        return layers
    
    def _build_pooling(self, pooling_config):
        layer = nn.ModuleList()
        params = {
            'embedding_dim': pooling_config.embedding_dim,
            'seq_fields': pooling_config.seq_fields,
            'seq_len': pooling_config.seq_len
        }
        for p in pooling_config.pooling_types:
            layer.append(POOLING_LIB[p](params))
            
        return layer

    def get_uniform_sample_arch(self, timeout=500):
        get_random_user_tower = lambda: tuple(
            np.random.randint(len(self._user_tower_config.blocks_types[i])) for i in range(self._user_tower_config.num_layers))
        get_random_item_tower = lambda: tuple(
            np.random.randint(len(self._item_tower_config.blocks_types[i])) for i in range(self._item_tower_config.num_layers))
        get_random_pooling = lambda: np.random.randint(len(self._pooling_config.pooling_types))

        for _ in range(timeout):
            return get_random_user_tower(), get_random_item_tower(), get_random_pooling()

        return get_random_user_tower(), get_random_item_tower(), get_random_pooling()


    def forward(self, users, histories, items, architecture, only_tower_output=False):
        users_embs = get_embeddings(self._users_embedding, users,
                                    multihot_fea=check_multihot_column(self._dataset_name, 'user'))
        items_embs = get_embeddings(self._items_embedding, items,
                                    multihot_fea=check_multihot_column(self._dataset_name, 'item'))
        
        histories_embs, mask = self.pooling_history(histories) # [B, L, field * E]
        
        user_tower_arch, item_tower_arch, pooling_type = architecture
        pooling = self._pooling_layer[pooling_type]
        
        histories_embs = pooling(histories_embs, mask)

        users_embs = torch.cat([users_embs, histories_embs], dim=1)
        
        users_embs_trans = self._users_trans(users_embs)
        items_embs_trans = self._items_trans(items_embs)
        
        user_tower_inputs = [users_embs_trans]
        for blocks, block_id in zip(self._user_tower, user_tower_arch):
            x = blocks[block_id](user_tower_inputs)
            user_tower_inputs.append(x)

        item_tower_inputs = [items_embs_trans]
        for blocks, block_id in zip(self._item_tower, item_tower_arch):
            x = blocks[block_id](item_tower_inputs)
            item_tower_inputs.append(x)

        user_output = F.normalize(user_tower_inputs[-1], p=2, dim=1)
        item_output = F.normalize(item_tower_inputs[-1], p=2, dim=1)

        if not only_tower_output:
            score = torch.matmul(user_output, item_output.T)
            regs = self._reg * (torch.norm(users_embs) + torch.norm(items_embs))
            return score, regs
        else:
            return user_output, item_output


class FixedSupernet(Base):
    def __init__(self,
                 dataset,
                 user_tower_config,
                 item_tower_config,
                 pooling_config,
                 arch):
        super(FixedSupernet, self).__init__(dataset=dataset,
                                            user_tower_config=user_tower_config,
                                            item_tower_config=item_tower_config,
                                            pooling_config=pooling_config)
        self._user_tower_arch, self._item_tower_arch, self._pooling_type = arch
        self._user_tower = self._build_tower(self._user_tower_config, self._user_tower_arch)
        self._item_tower = self._build_tower(self._item_tower_config, self._item_tower_arch)
        self._pooling_layer = self._build_pooling(self._pooling_config, self._pooling_type)

    def __repr__(self):
        user_tower = ', '.join([self._user_tower_config.blocks_types[i][j] for i, j in enumerate(self._user_tower_arch)])
        item_tower = ', '.join([self._item_tower_config.blocks_types[i][j] for i, j in enumerate(self._item_tower_arch)])
        pooling_layer = f'{self._pooling_config.pooling_types[self._pooling_type]}'
        return f"[({user_tower}), ({item_tower})], [{pooling_layer}]"

    @property
    def user_tower(self):
        return [self._user_tower_config.blocks_types[i][j] for i, j in enumerate(self._user_tower_arch)]

    @property
    def item_tower(self):
        return [self._item_tower_config.blocks_types[i][j] for i, j in enumerate(self._item_tower_arch)]
    
    @property
    def pooling_layer(self):
        return self._pooling_config.pooling_types[self._pooling_type]

    def get_tag(self):
        tag = super(FixedSupernet, self).get_tag()

        return f'fixed{SUPERNET_TAG}' + tag

    def _build_tower(self, tower_config, arch):
        layers = nn.ModuleList()
        num_inputs = 1
        for i in range(tower_config.num_layers):
            params = {
                'block_in_dim': tower_config.block_in_dim,
                'block_out_dim': tower_config.block_in_dim,
                'tower_out_dim': tower_config.tower_out_dim,
                'embedding_dim': tower_config.embedding_dim,
                'num_inputs': num_inputs,
                'is_first_layer': False,
                'is_last_layer': False
            }
            if i == 0:
                params['is_first_layer'] = True
            elif i == tower_config.num_layers - 1:
                params['is_last_layer'] = True
            layers.append(BLOCKS_LIB[tower_config.blocks_types[i][arch[i]]](params))
            num_inputs += 1

        return layers
    
    def _build_pooling(self, pooling_config, pooling_type):
        params = {
            'embedding_dim': pooling_config.embedding_dim,
            'seq_fields': pooling_config.seq_fields,
            'seq_len': pooling_config.seq_len
        }
        pooling_layer = POOLING_LIB[pooling_config.pooling_types[pooling_type]](params)
            
        return pooling_layer


    def forward(self, users, histories, items, only_tower_output=False):
        users_embs = get_embeddings(self._users_embedding, users,
                                    multihot_fea=check_multihot_column(self._dataset_name, 'user'))
        items_embs = get_embeddings(self._items_embedding, items,
                                    multihot_fea=check_multihot_column(self._dataset_name, 'item'))
        
        histories_embs, mask = self.pooling_history(histories) # [B, L, field * E]

        histories_embs = self._pooling_layer(histories_embs, mask)

        users_embs = torch.cat([users_embs, histories_embs], dim=1)
        
        users_embs_trans = self._users_trans(users_embs)
        items_embs_trans = self._items_trans(items_embs)

        user_tower_inputs = [users_embs_trans]
        for block in self._user_tower:
            x = block(user_tower_inputs)
            user_tower_inputs.append(x)

        item_tower_inputs = [items_embs_trans]
        for block in self._item_tower:
            x = block(item_tower_inputs)
            item_tower_inputs.append(x)

        user_output = F.normalize(user_tower_inputs[-1], p=2, dim=1)
        item_output = F.normalize(item_tower_inputs[-1], p=2, dim=1)

        if not only_tower_output:
            score = torch.matmul(user_output, item_output.T)
            regs = self._reg * (torch.norm(users_embs) + torch.norm(items_embs))
            return score, regs
        else:
            return user_output, item_output
