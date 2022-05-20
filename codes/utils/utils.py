#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import re
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys

from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from torch.backends import cudnn

from codes.blocks.block import BLOCK2ID, ID2BLOCK, BLOCK_FLOPS
from codes.blocks.pooling import POOLING2ID, ID2POOLING, POOLING_LIB
from codes.utils.torchprofile import profile_macs

PROJECT_PATH = os.path.abspath(os.path.join(__file__, '../../..'))
PROJECT_PARENT_PATH = os.path.abspath(os.path.join(os.path.dirname(PROJECT_PATH)))
# CONFIG_PATH = os.path.join(PROJECT_PATH, 'codes', 'configs', 'embedding_test')
CONFIG_PATH = os.path.join(PROJECT_PATH, 'codes', 'configs', 'layer_test')
SUPERNET_CHECKPOINT_PATH = 'supernet_checkpoint'
SEARCH_CHECKPOINT_PATH = 'search_checkpoint'
RETRAIN_CHECKPOINT_PATH = 'retrain_checkpoint'
BASELINE_TRAIN_CHECKPOINT = 'baseline_train_checkpoint'
SUPERNET_TAG = 'supernet'
# print(PROJECT_PARENT_PATH, PROJECT_PATH)
AUTOTOWER = 'AutoTower'

TOPKS = [1, 5, 10, 50, 100]


def get_logger(name, save_dir=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    # logger.setLevel(level)
    if save_dir is None:
        return logger
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)

    log_fmt = '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(stream=sys.stdout, level=level, format=log_fmt, datefmt=date_fmt)

    temp = os.path.split(save_dir)
    debug_save_dir = os.path.join(temp[0], 'debug_' + temp[1])
    fh_debug = logging.FileHandler(debug_save_dir)
    # fh_debug.setLevel(level)
    debug_filter = logging.Filter()
    debug_filter.filter = lambda record: record.levelno >= level
    fh_debug.addFilter(debug_filter)
    fh_debug.setFormatter(logging.Formatter(log_fmt, date_fmt))
    logger.addHandler(fh_debug)

    fh_info = logging.FileHandler(save_dir)
    # fh_info.setLevel(logging.INFO)
    info_filter = logging.Filter()
    info_filter.filter = lambda record: record.levelno >= logging.INFO
    fh_info.addFilter(info_filter)
    fh_info.setFormatter(logging.Formatter(log_fmt, date_fmt))
    logger.addHandler(fh_info)

    # fh_console = logging.StreamHandler()
    # fh_console.setLevel(logging.INFO)
    # fh_console.setFormatter(logging.Formatter(log_fmt, date_fmt))
    # logging.getLogger(name).addHandler(fh_console)
    # logger.addHandler(fh_console)

    return logger


def set_seed(seed=666, cudnn_benchmark=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = cudnn_benchmark


def check_directory(path, force_removed=False):
    if force_removed:
        try:
            shutil.rmtree(path)
        except Exception as e:
            pass

    if not os.path.exists(path):
        os.makedirs(path)


def check_file(path, force_removed=False):
    if force_removed:
        try:
            os.remove(path)
        except Exception as e:
            pass
    # if not os.path.exists(path):
    #     os.makedirs(path)


def linecount_wc(file):
    return int(os.popen(f'wc -l {file}').read().split()[0])


def create_exp_dir(path, scripts_to_save=None, force_removed=False):
    if force_removed:
        try:
            shutil.rmtree(path)
        except Exception as e:
            pass

    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups


def count_parameters_in_mb(model):
    return np.sum(p.numel() for name, p in model.named_parameters() if "auxiliary" not in name
                  and p is not None and p.requires_grad) / 1e6


def get_lastest_model(path, tag='None'):
    if not os.path.exists(path):
        os.makedirs(path)
    model_list = os.listdir(path)
    model_list = [m for m in model_list if tag in m]
    if not model_list:
        return None, 0
    model_list.sort()
    lastest_model = model_list[-1]
    iters = lastest_model.split('#')[-1]
    iters = int(re.findall(r'\d+', iters)[0])

    return os.path.join(path, lastest_model), iters


class EarlyStop:
    def __init__(self, patience=3, method='min'):
        self._metrics = []
        self._patience = patience
        self._not_rise_times = 0
        self._cur_max = None
        self._method = method

    @property
    def not_rise_times(self):
        return self._not_rise_times

    @not_rise_times.setter
    def not_rise_times(self, x):
        self._not_rise_times = x

    @property
    def cur_max(self):
        return self._cur_max

    @cur_max.setter
    def cur_max(self, x):
        self._cur_max = x

    def add_metric(self, m):
        if self._method == 'min':
            m = -m
        self._metrics.append(m)
        if self._cur_max is None:
            self._cur_max = m
        if m > self._cur_max:
            self._cur_max = m
            self._not_rise_times = 0
        else:
            self._not_rise_times += 1

        if self._not_rise_times >= self._patience:
            return True
        else:
            return False


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out



def encode_pooling_types(block_types):
    pooling2id = POOLING2ID
    res = 0
    for blk in block_types:
        if blk in pooling2id:
            id = pooling2id[blk]
            res ^= (1 << id)

    return res


def decode_pooling_types(code):
    id2pooing = ID2POOLING
    size = len(ID2POOLING.keys())
    blocks = []

    for i in range(size):
        if (code >> i) & 1 == 1:
            blocks.append(id2pooing[i])

    return blocks


def encode_block_types(block_types):
    block2id = BLOCK2ID
    res = 0
    for blk in block_types:
        if blk in block2id:
            id = block2id[blk]
            res ^= (1 << id)

    return res


def decode_block_types(code):
    id2block = ID2BLOCK
    size = len(ID2BLOCK.keys())
    blocks = []

    for i in range(size):
        if (code >> i) & 1 == 1:
            blocks.append(id2block[i])

    return blocks


def resolve_config(config_file_path):
    with open(config_file_path, 'r') as f:
        json_data = json.load(f)
    return json_data['num_layers'], json_data['embedding_dim'], json_data['block_in_dim'], json_data['tower_out_dim']


def initialize_embeddings(fea_dims, emb_dims, multihot_fea=-1):
    embedding_tables = nn.ModuleList()

    for i, d in enumerate(fea_dims):
        if i != multihot_fea:
            embedding_table = nn.Embedding(num_embeddings=d, embedding_dim=emb_dims[i])
        else:
            embedding_table = nn.Embedding(num_embeddings=d + 1, embedding_dim=emb_dims[i], padding_idx=d)
        torch.nn.init.xavier_uniform_(embedding_table.weight.data)
        embedding_tables.append(embedding_table)

    return embedding_tables


def get_embeddings(embedding_tables, x, multihot_fea=-1, is_concat=True):
    embeddings = []
    l = multihot_fea if multihot_fea != -1 else x.shape[1]
    for i in range(l):
        embedding = embedding_tables[i](x[:, i])
        embeddings.append(embedding)
    if multihot_fea != -1:
        size_tensor = x[:, multihot_fea] + 1e-9
        size_tensor = size_tensor.unsqueeze(1)
        embedding = embedding_tables[multihot_fea](x[:, multihot_fea + 1:])
        embedding = torch.sum(embedding, dim=1) / size_tensor
        embeddings.append(embedding)

    if is_concat:
        embeddings = torch.cat(embeddings, dim=1)

    return embeddings


def profile(model, *inputs, batch_size=1024):
    """profile model FLOPs and parameters
    """
    flops = profile_macs(model, *inputs)
    flops //= 2 * batch_size
    params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'embedding' not in name)
    return flops, params


def get_pooling_flops(pooling_type, embedding_dim, seq_fields, seq_len):
    """Calculate FLOPs and params of Pooling layer.

    Args:
        pooling_type (str or int): pooling type
        embedding_dim (int)
        seq_fields (int): number of sequence fields
        seq_len (int):
    """
    param = {
        'embedding_dim': embedding_dim,
        'seq_fields': seq_fields,
        'seq_len': seq_len
    }
    pooling = POOLING_LIB[pooling_type](param)
    
    fake_input = torch.randn(10, seq_len, embedding_dim * seq_fields)
    fake_mask = torch.ones((10, seq_len), dtype=torch.bool)
    
    flops, params = profile(pooling, fake_input, fake_mask, batch_size=10)
    
    return flops, params



def get_supernet_trans_flops(in_dim):
    """Calculate supernet transition layer's FLOPs and params

    Args:
        in_dim: input dim (embedding_dim * num_fields)
    """
    return in_dim * 32, (in_dim + 1) * 64


def get_cand_flops(arch, block_types, in_dim):
    total_flops, total_params = 0, 0
    for idx, blk in enumerate(arch):
        key = f'{idx}_{block_types[idx][blk]}'
        flops, params = BLOCK_FLOPS[key]
        total_flops += flops
        total_params += params

    # trans_flops, trans_params = get_supernet_trans_flops(ds, type)
    trans_flops, trans_params = get_supernet_trans_flops(in_dim)
    trans_flops += trans_flops
    trans_params += trans_params

    return total_flops, total_params


def get_model_flops(arch, ds):
    user_tower_flops, user_tower_params = 0, 0
    for idx, blk in enumerate(arch.user_tower):
        key = f'{idx}_{blk}'
        flops, params = BLOCK_FLOPS[key]
        user_tower_flops += flops
        user_tower_params += params
    trans_flops, trans_params = get_supernet_trans_flops(sum(arch._user_fea_emb_dims) + arch._seq_fields * \
        arch._item_embedding_dim)
    user_tower_flops += trans_flops
    user_tower_params += trans_params
    
    pooling_flops, pooling_params = get_pooling_flops(arch.pooling_layer, arch._item_embedding_dim,
                                                      arch._seq_fields, arch._seq_len)
    user_tower_flops += pooling_flops
    user_tower_params += pooling_params

    item_tower_flops, item_tower_params = 0, 0
    for idx, blk in enumerate(arch.item_tower):
        key = f'{idx}_{blk}'
        flops, params = BLOCK_FLOPS[key]
        item_tower_flops += flops
        item_tower_params += params
    trans_flops, trans_params = get_supernet_trans_flops(sum(arch._item_fea_emb_dims))
    item_tower_flops += trans_flops
    item_tower_params += trans_params

    total_flops = user_tower_flops + item_tower_flops
    total_params = user_tower_params + item_tower_params

    return user_tower_flops, user_tower_params, item_tower_flops, item_tower_params, total_flops, total_params
