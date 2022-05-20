#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import os
import torch.nn as nn
import torch

from codes.supernet.network import Supernet
from codes.utils.utils import get_lastest_model


class Searcher(nn.Module, metaclass=ABCMeta):
    def __init__(self, name, train_dataloader, val_dataloader, max_epochs, num_users, num_items, reg, args):
        super(Searcher, self).__init__()
        self._name = name
        self._args = args

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

        self._num_users = num_users
        self._num_items = num_items
        self._max_epochs = max_epochs

        self._reg = reg

        self._search_time = 0
        self._num_eval = 0
        self._epoch = 0

        self._logger = args.logger
        self._writter = args.writter

    def _get_supernet(self, load_supernet=False, only_kwargs=False):
        self._logger.info(f"load supernet from {self._args.supernet_checkpoint_path}/{self._args.supernet_tag}")
        lastest_model, iters = get_lastest_model(self._args.supernet_checkpoint_path, self._args.supernet_tag)
        self._logger.info(f"load {lastest_model} iters {iters}")
        checkpoint = torch.load(lastest_model)
        if only_kwargs:
            return checkpoint['kwargs']
        supernet = Supernet(**checkpoint['kwargs'])
        if self._args.use_gpu:
            supernet = supernet.cuda()
        # self._supernet = torch.nn.DataParallel(self._supernet)
        if self._args.use_pretrain_supernet and load_supernet:
            supernet.load_state_dict(checkpoint['state_dict'], strict=True)

        return supernet

    @property
    def checkpoint_file_path(self):
        checkpoint_path = self._args.search_checkpoint_path
        checkpoint_file = os.path.join(checkpoint_path, f'{self.get_tag()}_search-checkpoint.pth.tar')

        return checkpoint_file

    @staticmethod
    def new_retrain_model(dataset, checkpoint_path, topk=1):
        pass

    @abstractmethod
    def get_tag(self):
        pass

    @abstractmethod
    def _save_checkpoint(self, **kwargs):
        pass

    @abstractmethod
    def _load_checkpoint(self):
        pass

    @abstractmethod
    def search(self):
        pass

    @property
    def name(self):
        return self._name

    @property
    def search_time(self):
        return self._search_time

    @search_time.setter
    def search_time(self, x):
        self._search_time = x
