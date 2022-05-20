import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))

from codes.configs.config import CONFIG, POOLING_BLOCK

import traceback
import argparse
import logging
import time
import ast

import torch
import torch.nn
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from codes.search.evolution_searcher import EvolutionSearcher
from codes.datasets.dataset import DataProvider, get_dataset
from codes.supernet.network import Supernet
from codes.utils.utils import PROJECT_PATH, check_file, set_seed, check_directory, get_logger, AUTOTOWER, get_parameters, \
    count_parameters_in_mb, get_lastest_model, PROJECT_PARENT_PATH, EarlyStop, SUPERNET_CHECKPOINT_PATH, \
    SEARCH_CHECKPOINT_PATH, RETRAIN_CHECKPOINT_PATH, CONFIG_PATH, resolve_config, TOPKS, get_model_flops
from codes.utils.evaluate import evaluate, supernet_evaluate, get_recall_ratio, get_recall_ncdg_mrr, compare_recall_ratio_list


def get_args():
    parser = argparse.ArgumentParser()
    # TODO: need to check
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--valid_ratio', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=1024)

    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adagrad', 'adam'])
    # for adam
    parser.add_argument('--adam_lr', type=float, default=0.001)
    parser.add_argument('--adam_weight_decay', type=float, default=1e-6)
    # for Adagrad
    parser.add_argument('--adagrad_lr', type=float, default=5e-2, help='init learning rate')

    # for supernet
    parser.add_argument('--use_pretrain_supernet', type=ast.literal_eval, default=True)
    # parser.add_argument('--total_iters', type=int, default=150000, help='total iters')
    parser.add_argument('--total_iters', type=int, default=50000, help='total iters')
    parser.add_argument('--val_iters', type=int, default=10000, help='total iters')
    parser.add_argument('--auto_continue', type=ast.literal_eval, default=True, help='report frequency')
    parser.add_argument('--display_interval', type=int, default=5000, help='report frequency')
    parser.add_argument('--save_interval', type=int, default=10000, help='report frequency')

    # for search
    parser.add_argument('--search_patience', type=int, default=5)

    # for AUTOTOWER
    # parser.add_argument('--search_epochs', type=int, default=100)
    parser.add_argument('--search_epochs', type=int, default=50)
    parser.add_argument('--select_num', type=int, default=10)
    parser.add_argument('--population_num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover_num', type=int, default=25)
    parser.add_argument('--mutation_num', type=int, default=25)
    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--max_test_iters', type=int, default=40)
    ## complexity constrain
    parser.add_argument('--use_flops_limits', type=ast.literal_eval, default=True)
    parser.add_argument('--use_total_flops_limits', type=ast.literal_eval, default=True)
    # parser.add_argument('--total_flops_limit', type=float, default=1701576704)
    parser.add_argument('--total_flops_limit', type=float, default=415002600)
    parser.add_argument('--user_tower_flops_limit', type=float, default=850788352)
    parser.add_argument('--item_tower_flops_limit', type=float, default=850788352)
    parser.add_argument('--use_params_limits', type=ast.literal_eval, default=False)
    parser.add_argument('--use_total_params_limits', type=ast.literal_eval, default=False)
    parser.add_argument('--total_params_limit', type=float, default=1665152)
    parser.add_argument('--user_tower_params_limit', type=float, default=832576)
    parser.add_argument('--item_tower_params_limit', type=float, default=832576)

    # for rl
    # parser.add_argument('--rl_search_epochs', type=int, default=100, help='num of searching epochs')
    parser.add_argument('--rl_search_epochs', type=int, default=50, help='num of searching epochs')
    parser.add_argument('--rl_controller_lr', type=float, default=3.5e-4)
    parser.add_argument('--rl_controller_hid', type=int, default=100)
    parser.add_argument('--rl_softmax_temperature', type=float, default=5.0)
    parser.add_argument('--rl_tanh_c', type=float, default=2.5)
    parser.add_argument('--rl_shared_initial_step', type=int, default=100)
    parser.add_argument('--rl_shared_num_sample', type=int, default=1)
    parser.add_argument('--rl_controller_max_step', type=int, default=100)
    parser.add_argument('--rl_entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--rl_discount', type=float, default=1.0)
    parser.add_argument('--rl_controller_grad_clip', type=float, default=0)
    parser.add_argument('--rl_ema_baseline_decay', type=float, default=0.95)
    # for nasp
    # parser.add_argument('--nasp_search_epochs', type=int, default=100, help='num of searching epochs')
    parser.add_argument('--nasp_search_epochs', type=int, default=50, help='num of searching epochs')
    parser.add_argument('--nasp_arch_lr', type=float, default=0.001)
    parser.add_argument('--nasp_arch_weight_decay', type=float, default=1e-6)
    parser.add_argument('--nasp_sgd_lr', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--nasp_sgd_lr_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--nasp_sgd_momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--nasp_sgd_weight_decay', type=float, default=3e-4)
    parser.add_argument('--nasp_sgd_gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--nasp_sgd_decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--nasp_e_greedy', type=float, default=0)

    # for retrain
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--retrain_max_epochs', type=int, default=100)

    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--parallel_num', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--searcher', type=str, default='AUTOTOWER', choices=['AUTOTOWER', 'random', 'sif', 'rl', 'nasp'])
    parser.add_argument('--dataset', type=str, default='ml100k', choices=['ml100k', 'ml1m', 'amazon', 'frappe', 'yelp'])
    parser.add_argument('--dataset_path', type=str, default=f'{PROJECT_PATH}/codes/datasets/data/ml100k/ml100k.txt')
    parser.add_argument('--log_save_dir', type=str, default=PROJECT_PATH)
    parser.add_argument('--supernet_tag', type=str, default='supernet_5_64_20_20_1e-05_16252879_524288')
    parser.add_argument('--supernet_checkpoint_path', type=str,
                        default=os.path.join(PROJECT_PARENT_PATH, SUPERNET_CHECKPOINT_PATH))
    parser.add_argument('--search_checkpoint_path', type=str,
                        default=os.path.join(PROJECT_PARENT_PATH, SEARCH_CHECKPOINT_PATH))
    parser.add_argument('--retrain_checkpoint_path', type=str,
                        default=os.path.join(PROJECT_PARENT_PATH, RETRAIN_CHECKPOINT_PATH))

    parser.add_argument('--use_pretrained_embeddings', type=ast.literal_eval, default=False)
    parser.add_argument('--use_gpu', type=ast.literal_eval, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--show_tfboard', type=ast.literal_eval, default=False)
    
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--seq_fields', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=16)

    args = parser.parse_args()

    if 'ml100k' in args.dataset_path:
        args.dataset = 'ml100k'
    elif 'ml1m' in args.dataset_path:
        args.dataset = 'ml1m'
    elif 'frappe' in args.dataset_path:
        args.dataset = 'frappe'
    elif 'amazon_beauty' in args.dataset_path:
        args.dataset = 'amazon_beauty'
    elif 'amazon_fashion' in args.dataset_path:
        args.dataset = 'amazon_fashion'
    elif 'amazon_book' in args.dataset_path:
        args.dataset = 'amazon_book'
    elif 'yelp' in args.dataset_path:
        args.dataset = 'yelp'
    else:
        raise Exception(f"No such dataset {args.dataset_path}")

    return args


def do_train_supernet(args, user_tower_config, item_tower_config, pooling_config, dataset):
    logger = args.logger
    # writter = args.writter
    logger.info(f'ARGS {args}')
    set_seed(args.seed)

    t = time.time()
    train_dataloader, val_dataloader, _ = dataset.get_dataloader(batch_size=args.batch_size)
    train_provider = DataProvider(train_dataloader)
    logger.info(f"prepare data finish, use_time={time.time() - t}")

    logger.info(f"start training supernet...")
    supernet = Supernet(dataset=dataset,
                        user_tower_config=user_tower_config,
                        item_tower_config=item_tower_config,
                        pooling_config=pooling_config)
    logger.info(f"supernet tag: {supernet.get_tag()}")
    if args.use_gpu:
        supernet = supernet.cuda()

    if args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(get_parameters(supernet), args.adagrad_lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(get_parameters(supernet), args.adam_lr, weight_decay=args.adam_weight_decay)
    else:
        raise Exception(f"No such optimizer {args.optimizer}!")
    logger.info(f"search total param size={count_parameters_in_mb(supernet)}MB")

    cur_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model(args.supernet_checkpoint_path, tag=supernet.get_tag())
        if lastest_model is not None:
            logger.info(f"load supernet chechpoint from {lastest_model}, checkpoint iter: {iters}")
            cur_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if args.use_gpu else 'cpu')
            train_use_time = checkpoint['train_use_time']
            supernet.train_use_time = train_use_time
            supernet.load_state_dict(checkpoint['state_dict'], strict=True)
            supernet._A = checkpoint['correction_dict_A']
            supernet._B = checkpoint['correction_dict_B']
            logger.info(f'load latest model')
            # for i in range(iters):
            #     scheduler.step()

    while cur_iters < args.total_iters:
        t = time.time()
        cur_iters = train(supernet=supernet,
                          train_provider=train_provider,
                          optimizer=optimizer,
                          cur_iters=cur_iters,
                          val_iters=args.val_iters,
                          args=args)
        cur_iter_train_use_time = time.time() - t
        supernet.train_use_time += cur_iter_train_use_time

        t = time.time()
        val_loss, val_auc, val_hr_topk = supernet_evaluate(supernet, val_dataloader, args.use_gpu, cal_hr=False, topk=TOPKS)
        val_hr_topk_str = ', '.join([f'val_hr{k}={v}' for k, v in val_hr_topk.items()])

        cur_iter_evaluate_use_time = time.time() - t

        logger.info(f"run {cur_iters}/{args.total_iters} iters, "
                    f"val_loss={val_loss}, "
                    f"val_auc={val_auc},"
                    f"{val_hr_topk_str}, "
                    f"cur_iters_evaluate_use_time={cur_iter_evaluate_use_time:.6f}, "
                    f"cur_iters_train_use_time={cur_iter_train_use_time:.6f}, "
                    f"total_iters_train_use_time={supernet.train_use_time:.6f}")

    logger.info(f'finish train supernet, use_time={supernet.train_use_time}')


def train(supernet: Supernet, train_provider, optimizer, cur_iters, val_iters, args, tqdm_interval=10):
    logger = args.logger
    writter = args.writter
    tt = tqdm(range(val_iters), smoothing=0, mininterval=1.0)
    total_train_loss = 0
    supernet.train()
    for iters in tt:
        cur_iters += 1

        t2 = time.time()
        users, histories, items, labels = train_provider.next()
        if args.use_gpu:
            users, histories, items, labels = users.cuda(), histories.cuda(), items.cuda(), labels.cuda()
        data_time = time.time() - t2

        t3 = time.time()
        sample_arch = supernet.get_uniform_sample_arch()
        preds, regs = supernet(users, histories, items, sample_arch)
        # loss = supernet.compute_loss(preds, labels, regs)
        loss = supernet.compute_loss_corrected(preds, labels, regs, items, cur_iters)
        optimizer.zero_grad()
        loss.backward()
        # for p in model.parameters():
        #     if p.grad is not None and p.grad.sum() == 0:
        #         p.grad = None
        torch.nn.utils.clip_grad_norm_(supernet.parameters(), args.grad_clip)
        optimizer.step()
        train_time = time.time() - t3
        loss = loss.item()
        total_train_loss += loss

        if iters % tqdm_interval == 0:
            tt.set_postfix(train_loss=total_train_loss / tqdm_interval)
            total_train_loss = 0
        if cur_iters % args.display_interval == 0:
            print_info = f"train_iter={cur_iters}, loss={loss:.6f}"
            logger.info(print_info)
        if cur_iters % args.save_interval == 0:
            supernet.save_checkpoint(args.supernet_checkpoint_path, cur_iters)
        # if args.show_tfboard:
        #     writter.add_scalar('train_supernet/loss/train', loss, cur_iters)
        #     writter.add_scalar('train_supernet/regs/train', regs.item(), cur_iters)

    return cur_iters


def train_supernet():
    args = get_args()
    set_seed(args.seed)
    seed = args.seed

    supernet_checkpoint_path = args.supernet_checkpoint_path

    pooling_config = POOLING_BLOCK
    pooling_config['embedding_dim'] = args.embedding_dim
    pooling_config['seq_fields'] = args.seq_fields
    pooling_config['seq_len'] = args.seq_len
    
    item_tower_config = user_tower_config = CONFIG.copy()
    item_tower_config['embedding_dim'] = args.embedding_dim
    user_tower_config['embedding_dim'] = args.embedding_dim
    
    args.supernet_checkpoint_path = os.path.join(supernet_checkpoint_path, args.dataset, str(args.seed))

    log_root_path = os.path.join(PROJECT_PATH, 'logs', 'supernet', args.dataset, str(seed),
                                    f"{user_tower_config['num_layers']}_{user_tower_config['embedding_dim']}_"
                                    f"{user_tower_config['block_in_dim']}_{user_tower_config['tower_out_dim']}_{user_tower_config['reg']}"
                                    f"{pooling_config['seq_fields']}_{pooling_config['seq_len']}")
    check_directory(log_root_path, force_removed=False)
    log_save_dir = os.path.join(log_root_path, f'log_supernet_{args.dataset}_'
                                                f'{args.total_flops_limit}_{args.user_tower_flops_limit}_{args.item_tower_flops_limit}_'
                                                f'{args.total_params_limit}_{args.user_tower_params_limit}_{args.item_tower_params_limit}_'
                                                f'{str(args.reg)}_{str(args.seed)}.txt')
    check_file(log_save_dir, force_removed=True)
    logger = get_logger(log_save_dir, log_save_dir, level=logging.DEBUG)
    args.logger = logger
    args.writter = None
    # if args.show_tfboard:
    #     tf_board_file = os.path.join(log_root_path, f'tfboard_{tag}')
    #     check_directory(tf_board_file, force_removed=True)
    #     writter = SummaryWriter(tf_board_file)
    #     args.writter = writter
    # dataset
    dataset = get_dataset(args, 'AUTOTOWRE', logger=logger)
    try:
        do_train_supernet(args, user_tower_config, item_tower_config, pooling_config, dataset)
    except Exception as e:
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    train_supernet()
