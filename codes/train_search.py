import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))

from codes.configs.config import CONFIG, POOLING_BLOCK, make_tag

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
from codes.utils.utils import PROJECT_PATH, set_seed, check_file, check_directory, get_logger, AUTOTOWER, get_parameters, \
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
    
    parser.add_argument('--need_research', type=ast.literal_eval, default=False)

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
    parser.add_argument('--use_flops_limits', type=ast.literal_eval, default=False)
    parser.add_argument('--use_total_flops_limits', type=ast.literal_eval, default=False)
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
        args.dataset = 'amazon_book'
    elif 'yelp' in args.dataset_path:
        args.dataset = 'yelp'
    else:
        raise Exception(f"No such dataset {args.dataset_path}")

    return args


def do_search(args, dataset):
    logger = args.logger
    writter = args.writter
    logger.info(f'ARGS {args}')
    cudnn.benchmark = True
    t = time.time()
    train_dataloader, val_dataloader, _ = dataset.get_dataloader(batch_size=args.batch_size)
    logger.info(f"prepare data finish, use_time={time.time() - t}")

    logger.info(f"start searching arch...")
    searcher = EvolutionSearcher(train_dataloader, val_dataloader, dataset.num_users, dataset.num_items, args)
    searcher.search()
    logger.info(f'total_search_time={searcher.search_time}')

    return searcher.checkpoint_file_path, searcher.search_time


def get_retrain_model(searcher_name, dataset, checkpoint_path, args):
    topk = args.topk
    cands = EvolutionSearcher.new_retrain_model(args, dataset, checkpoint_path, topk)

    return topk, cands


def retrain(args, dataset, search_checkpoint_path, search_time, tqdm_interval=10):
    logger = args.logger
    writter = args.writter
    logger.info(f'ARGS {args}')
    # checkpoint_file = os.path.join(args.search_checkpoint_path, 'search_checkpoint.pth.tar')
    # if not os.path.exists(checkpoint_file):
    #     return False
    # checkpoint = torch.load(checkpoint_file)

    train_queue, val_queue, test_queue = dataset.get_dataloader(batch_size=args.batch_size)
    topk, cands = get_retrain_model(args.searcher, dataset, search_checkpoint_path, args)
    logger.info(f'start to retrain top f{topk} cands.')
    best_test_loss = float('inf')
    best_test_auc = -float('inf')
    best_test_hr_topk = None
    best_flops_info = {}
    topks = TOPKS
    best_test_recall_ratio_topk = None
    best_test_ndcg_topk = None
    best_test_mrr_topk = None
    best_model = None
    for idx, (model, last_val_loss) in enumerate(cands):
        if args.use_gpu:
            model = model.cuda()
        flops_info = {}
        flops_info['user_tower_flops'], flops_info['user_tower_params'],\
        flops_info['item_tower_flops'], flops_info['item_tower_params'],\
        flops_info['total_flops'], flops_info['total_params'] = get_model_flops(model, args.dataset)
        logger.info(f"start retraining {model}, last_val_loss={last_val_loss}")
        early_stopper = EarlyStop(patience=args.patience)
        cur_best = float('inf')
        model_save_path = os.path.join(args.retrain_checkpoint_path)
        # TODO: whether use retrain checkpont - default no
        check_directory(model_save_path, force_removed=True)
        model_save_file = os.path.join(model_save_path, f'{model.get_tag()}-model-{str(idx)}.pth.tar')
        # TODO: remember to delete - used to skip evaluated models
        if os.path.exists(model_save_file):
            logger.info('skip this model, since it has been evaluated.')
            continue
        correction_dict_save_file = os.path.join(model_save_path, f'{model.get_tag()}-correction-{str(idx)}.pkl')
        step = 0
        for epoch in range(args.retrain_max_epochs):
            if args.optimizer == 'adagrad':
                # TODO whether use get_parameters
                optimizer = torch.optim.Adagrad(model.parameters(), args.adagrad_lr)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), args.adam_lr, weight_decay=args.adam_weight_decay)
            else:
                raise Exception(f'No such optimizer {args.optimizer}')
            model.train()
            tt = tqdm(train_queue, smoothing=0, mininterval=1.0)
            total_train_loss = 0
            for i, (users, histories, items, labels) in enumerate(tt):
                step += 1
                if args.use_gpu:
                    users, histories, items, labels = users.cuda(), histories.cuda(), items.cuda(), labels.cuda()
                preds, regs = model(users, histories, items)
                # loss = model.compute_loss(preds, labels, regs)
                loss = model.compute_loss_corrected(preds, labels, regs, items, step)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                loss = loss.item()
                total_train_loss += loss

                if i % tqdm_interval == 0:
                    tt.set_postfix(train_loss=total_train_loss / tqdm_interval)
                    total_train_loss = 0
            val_loss, val_auc, val_hr_topk, _, _ = evaluate(model, val_queue, args.use_gpu, cal_flops=False, cal_hr=False)
            val_hr_topk_str = ', '.join([f'val_hr{k}={v}' for k, v in val_hr_topk.items()])
            if val_loss < cur_best:
                cur_best = val_loss
                torch.save(model.state_dict(), model_save_file)
                model.save_correction_dict(correction_dict_save_file)
            logger.info(f"epoch={epoch}, "
                        f"val_loss={val_loss}, best_val_loss={cur_best}, val_auc={val_auc}, {val_hr_topk_str}")
            is_stop = early_stopper.add_metric(val_loss)
            if is_stop:
                logger.info(f"Not rise for {args.patience}, stop train")
                break
        temp = torch.load(model_save_file, map_location=None if args.use_gpu else 'cpu')
        model.load_state_dict(temp, strict=True)
        model.load_correction_dict(correction_dict_save_file)
        test_loss, test_auc, test_hr_topk, params_profile, flops_profile = evaluate(model, test_queue, args.use_gpu,
                                                                            cal_flops=True, topk=TOPKS)  # TODO remember to delete
        params_with_emb = count_parameters_in_mb(model)
        # test_recall_ratio = get_recall_ratio(model, dataset, topks=topks, use_gpu=args.use_gpu)
        test_recall_ratio, test_ndcg, test_mrr = get_recall_ncdg_mrr(model, dataset, topks=topks, use_gpu=args.use_gpu)
        flops_info['flops_profile'] = flops_profile
        flops_info['params_profile'] = params_profile
        flops_info['params_with_emb'] = params_with_emb
        if compare_recall_ratio_list(best_test_recall_ratio_topk, test_recall_ratio, keys=[5, 50, 100]):
            best_test_loss = test_loss
            best_test_auc = test_auc
            best_test_hr_topk = test_hr_topk
            best_test_recall_ratio_topk = test_recall_ratio
            best_test_ndcg_topk = test_ndcg
            best_test_mrr_topk = test_mrr
            best_model = model
            best_flops_info = flops_info

        test_hr_topk_str = ', '.join([f'test_hr{k}={v}' for k, v in test_hr_topk.items()])
        test_recall_ratio_topk_str = ', '.join([f'recall_ratio{k}={v}' for k, v in test_recall_ratio.items()])
        test_ndcg_str = ', '.join([f'NDCG{k}={v}' for k, v in test_ndcg.items()])
        test_mrr_str = ', '.join([f'MRR{k}={v}' for k, v in test_mrr.items()])
        flops_info_str = ', '.join([f'{k}={v}' for k, v in flops_info.items()])
        logger.info(f'{model}, test_loss={test_loss}, best_test_loss={best_test_loss}, '
                    f'test_auc={test_auc}, '
                    f'{test_hr_topk_str}, '
                    f'{test_recall_ratio_topk_str}, '
                    f'{test_ndcg_str}, '
                    f'{test_mrr_str}, '
                    f'{flops_info_str}')
    best_test_hr_topk_str = ', '.join([f'test_hr{k}={v}' for k, v in best_test_hr_topk.items()])
    best_test_recall_ratio_topk_str = ', '.join([f'recall_ratio{k}={v}' for k, v in best_test_recall_ratio_topk.items()])
    best_test_ndcg_topk_str = ', '.join([f'NDCG{k}={v}' for k, v in best_test_ndcg_topk.items()])
    best_test_mrr_topk_str = ', '.join([f'MRR{k}={v}' for k, v in best_test_mrr_topk.items()])
    best_flops_info_str = ', '.join([f'{k}={v}' for k, v in best_flops_info.items()])
    logger.info(f'best_model={best_model}, test_loss={best_test_loss}, '
                f'test_auc={best_test_auc}, '
                f'{best_test_hr_topk_str}, '
                f'{best_test_recall_ratio_topk_str}, '
                f'{best_test_ndcg_topk_str}, '
                f'{best_test_mrr_topk_str}, '
                f'{best_flops_info_str}')



def search():
    args = get_args()
    set_seed(args.seed)

    supernet_checkpoint_path = args.supernet_checkpoint_path
    search_checkpoint_path = args.search_checkpoint_path
    retrain_checkpoint_path = args.retrain_checkpoint_path

    pooling_config = POOLING_BLOCK
    pooling_config['embedding_dim'] = args.embedding_dim
    pooling_config['seq_fields'] = args.seq_fields
    pooling_config['seq_len'] = args.seq_len
    
    num_layers, embedding_dim, block_in_dim, tower_out_dim, reg = args.num_layers, args.embedding_dim,  \
        args.block_in_dim, args.tower_out_dim, args.reg
    args.reg = reg
    args.supernet_checkpoint_path = os.path.join(supernet_checkpoint_path, args.dataset, str(args.seed))
    args.search_checkpoint_path = os.path.join(search_checkpoint_path, args.searcher,
                                                f'{args.use_flops_limits}_{args.total_flops_limit}_{args.use_params_limits}_{args.total_params_limit}',
                                                args.dataset, str(args.seed),
                                                f'{num_layers}_{embedding_dim}_{block_in_dim}_{tower_out_dim}_{str(reg)}')
    args.retrain_checkpoint_path = os.path.join(retrain_checkpoint_path, args.searcher,
                                                f'{args.use_flops_limits}_{args.total_flops_limit}_{args.use_params_limits}_{args.total_params_limit}',
                                                args.dataset, str(args.seed),
                                                f'{num_layers}_{embedding_dim}_{block_in_dim}_{tower_out_dim}_{str(reg)}')
    
    # re-search
    if args.need_research:
        check_directory(args.search_checkpoint_path, force_removed=True)
        check_directory(args.retrain_checkpoint_path, force_removed=True)
    
    # logging
    log_root_path = os.path.join(PROJECT_PATH, 'logs', 'search',
                                    f'{args.use_flops_limits}_{args.total_flops_limit}_{args.use_params_limits}_{args.total_params_limit}',
                                    args.dataset, str(args.seed),
                                    f'{num_layers}_{embedding_dim}_{block_in_dim}_{tower_out_dim}_{str(reg)}'
                                    f"{pooling_config['seq_fields']}_{pooling_config['seq_len']}", args.searcher)
    check_directory(log_root_path, force_removed=True)
    log_save_file = os.path.join(log_root_path,
                                    f'log_search_{args.dataset}_{num_layers}_{embedding_dim}_{block_in_dim}_{tower_out_dim}_{str(reg)}'
                                    f'_{args.searcher}_{args.seed}.txt')
    check_file(log_save_file, force_removed=True)
    logger = get_logger(log_save_file, log_save_file, level=logging.DEBUG)
    args.logger = logger
    args.writter = None
    if args.show_tfboard:
        writter = SummaryWriter(
            os.path.join(log_root_path, f'tfboard_{args.dataset}_{args.searcher}_{args.seed}'))
        args.writter = writter
    # dataset
    dataset = get_dataset(args, args.searcher, logger=logger)
    try:
        checkpoint_file_path, search_time = do_search(args, dataset)
        retrain(args, dataset, checkpoint_file_path, search_time)
    except Exception as e:
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    search()
