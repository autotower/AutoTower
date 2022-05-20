#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import multiprocessing
import os
import pickle
import random
import shutil

import numpy as np
import torch
# from prefetch_generator import BackgroundGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from codes.utils.utils import get_logger, AUTOTOWER, set_seed

# PRE_DENSE_FEATES_FILE = 'pre_users.npy'
# PRE_SPARSE_FEATES_FILE = 'pre_items.npy'
# LABELS_FEATES_FILE = 'labels.npy'

USERS_TRAIN_FILE = 'splits/users_train.npy'
HISTORIES_TRAIN_FILE = 'splits/histories_train.npy'
ITEMS_TRAIN_FILE = 'splits/items_train.npy'
LABEL_TRAIN_FILE = 'splits/labels_train.npy'
USERS_TRAIN_NEGA_FILE = 'splits/users_nega_train.npy'
HISTORIES_TRAIN_NEGA_FILE = 'splits/histories_nega_train.npy'
ITEMS_TRAIN_NEGA_FILE = 'splits/items_nega_train.npy'
LABEL_TRAIN_NEGA_FILE = 'splits/labels_nega_train.npy'
USERS_VAL_FILE = 'splits/users_val.npy'
HISTORIES_VAL_FILE = 'splits/histories_val.npy'
ITEMS_VAL_FILE = 'splits/items_val.npy'
LABEL_VAL_FILE = 'splits/labels_val.npy'
USERS_VAL_NEGA_FILE = 'splits/users_nega_val.npy'
HISTORIES_VAL_NEGA_FILE = 'splits/histories_nega_val.npy'
ITEMS_VAL_NEGA_FILE = 'splits/items_nega_val.npy'
LABEL_VAL_NEGA_FILE = 'splits/labels_nega_val.npy'
USERS_TEST_FILE = 'splits/users_test.npy'
HISTORIES_TEST_FILE = 'splits/histories_test.npy'
ITEMS_TEST_FILE = 'splits/items_test.npy'
LABEL_TEST_FILE = 'splits/labels_test.npy'
USER_RECALL_TEST_FILE = 'splits/user_recall_test.npy'
HISTORIES_RECALL_TEST_FILE = 'splits/histories_recall_test.npy'
ITEM_RECALL_TEST_FILE = 'splits/item_recall_test.npy'
INFO_FILE = 'splits/infos.pickle'
USER_2_ITEM_DIC_FILE = 'splits/user_2_item_dict.pickle'
USER_2_ITEM_TRAIN_DIC_FILE = 'splits/user_2_item_train_dict.pickle'
USER_2_ITEM_TEST_DIC_FILE = 'splits/user_2_item_test_dict.pickle'
TEST_USERS_FILE = 'splits/test_users.pickle'
TEST_ITEMS_FILE = 'splits/test_items.pickle'
ALL_ITEMS_FILE = 'splits/all_items.pickle'

FILES = [USERS_TRAIN_FILE, ITEMS_TRAIN_FILE, LABEL_TRAIN_FILE, HISTORIES_TRAIN_FILE,
         USERS_TRAIN_NEGA_FILE, ITEMS_TRAIN_NEGA_FILE, LABEL_TRAIN_NEGA_FILE, HISTORIES_TRAIN_NEGA_FILE,
         USERS_VAL_FILE, ITEMS_VAL_FILE, LABEL_VAL_FILE, HISTORIES_VAL_FILE,
         USERS_VAL_NEGA_FILE, ITEMS_VAL_NEGA_FILE, LABEL_VAL_NEGA_FILE, HISTORIES_VAL_NEGA_FILE,
         USERS_TEST_FILE, ITEMS_TEST_FILE, LABEL_TEST_FILE, HISTORIES_TEST_FILE,
         USER_RECALL_TEST_FILE, HISTORIES_RECALL_TEST_FILE, ITEM_RECALL_TEST_FILE,
         INFO_FILE,
         # USER_2_ITEM_DIC_FILE,
         USER_2_ITEM_TEST_DIC_FILE, TEST_ITEMS_FILE,
         ALL_ITEMS_FILE,  # all times
         USER_2_ITEM_TRAIN_DIC_FILE,
         TEST_USERS_FILE]

PRETRAINED_EMBEDDINGS_DIR = 'pretrained_embeddings'
BEST_PRETRAINED_EMBEDDINGS_FILE = 'pretrained_embeddings_best.pt'
FINAL_PRETRAINED_EMBEDDINGS_FILE = 'pretrained_embeddings_final.pt'



def check_multihot_column(dataset_name, emb_type):
    if dataset_name in ['ml100k', 'ml1m'] and emb_type == 'item':
        return 2

    return -1


def get_dataset(args, model_name, logger=None):
    negative_sample = 0

    if 'ml100k' in args.dataset_path:
        args.dataset = 'ml100k'
        from codes.datasets.movielens100k import Movielens100kDataset
        dataset = Movielens100kDataset(args.dataset_path, negative_sample=negative_sample, repreprocess=False, logger=logger)
    elif 'ml1m' in args.dataset_path:
        args.dataset = 'ml1m'
        from codes.datasets.movielens1m import Movielens1mDataset
        dataset = Movielens1mDataset(args.dataset_path, negative_sample=negative_sample, repreprocess=False, logger=logger)
    elif 'amazon_book' in args.dataset_path:
        args.dataset = 'amazon_book'
        from codes.datasets.amazon_book import AmazonBookDataset
        dataset = AmazonBookDataset(args.dataset_path, negative_sample=negative_sample, repreprocess=False,
                                    logger=logger)
    elif 'yelp' in args.dataset_path:
        args.dataset = 'yelp'
        from codes.datasets.yelp import YelpDataset
        dataset = YelpDataset(args.dataset_path, negative_sample=negative_sample, repreprocess=False,
                              logger=logger)
    else:
        raise Exception(f"No such dataset {args.dataset_path}")

    return dataset


class MyDataset(data.Dataset):
    def __init__(self, users, histories, items, label):
        self._users = users
        self._histories = histories
        self._items = items
        self._label = label
        self._num_data = len(self._label)

    def __getitem__(self, index):
        return self._users[index], self._histories[index], self._items[index], self._label[index]

    def __len__(self):
        return self._num_data

    @property
    def num_data(self):
        return self._num_data


class DataProvider(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1], data[2], data[3]


class Dataset(object):
    def __init__(self, name, data_path, num_user_feas, num_item_feas, repreprocess=False, val_ratio=0.25, test_ratio=0.25,
                 use_pretrained_embeddings=True, with_head=False, logger=None, seed=666, negative_sample=0):
        set_seed(seed=seed)
        self._name = name
        self._seed = seed
        self._data_path = data_path
        self._num_user_feas = num_user_feas
        self._user_fea_fields = None
        self._num_item_feas = num_item_feas
        self._item_fea_fields = None
        self._save_path = os.path.dirname(data_path)
        self._repreprocess = repreprocess
        self._with_head = with_head
        self._use_pretrained_embeddings = use_pretrained_embeddings
        self._pretrained_embeddings = None
        self._negative_sample = negative_sample
        self._user_fea_emb_dim = None
        self._item_fea_emb_dim = None

        self._num_users = None
        self._num_items = None
        self._num_data = None
        self._num_train_data = None
        self._num_val_data = None
        self._num_test_data = None
        self._train_ratio = 1 - val_ratio - test_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._users, self._histories, self._items, self._labels = None, None, None, None
        self._user_2_item_test_dic = None
        self._user_2_item_train_dic = None
        self._user_2_item_dic = None
        self._test_users, self._test_items = None, None

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        self._all_items = None  # collect all items

        if logger is None:
            self._logger = get_logger(AUTOTOWER)
        else:
            self._logger = logger

    def get_dataloader(self, batch_size, num_workers=1):
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
            self._logger.info(f"Cpu count {num_workers}")

        train_dataloader = DataLoader(self._train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        val_dataloader = DataLoader(self._val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        test_dataloader = DataLoader(self._test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        return train_dataloader, val_dataloader, test_dataloader

    def _read(self):
        print('read start')
        if self._check_preprocess_file():
            if self._negative_sample == 0:
                users_train = np.load(os.path.join(self._save_path, USERS_TRAIN_FILE), allow_pickle=True)
                histories_train = np.load(os.path.join(self._save_path, HISTORIES_TRAIN_FILE), allow_pickle=True)
                items_train = np.load(os.path.join(self._save_path, ITEMS_TRAIN_FILE), allow_pickle=True)
                labels_train = np.load(os.path.join(self._save_path, LABEL_TRAIN_FILE)).astype(np.float32)
                users_val = np.load(os.path.join(self._save_path, USERS_VAL_FILE), allow_pickle=True)
                histories_val = np.load(os.path.join(self._save_path, HISTORIES_VAL_FILE), allow_pickle=True)
                items_val = np.load(os.path.join(self._save_path, ITEMS_VAL_FILE), allow_pickle=True)
                labels_val = np.load(os.path.join(self._save_path, LABEL_VAL_FILE)).astype(np.float32)
            else:
                users_train = np.load(os.path.join(self._save_path, USERS_TRAIN_NEGA_FILE), allow_pickle=True)
                histories_train = np.load(os.path.join(self._save_path, HISTORIES_TRAIN_NEGA_FILE), allow_pickle=True)
                items_train = np.load(os.path.join(self._save_path, ITEMS_TRAIN_NEGA_FILE), allow_pickle=True)
                labels_train = np.load(os.path.join(self._save_path, LABEL_TRAIN_NEGA_FILE)).astype(np.float32)
                users_val = np.load(os.path.join(self._save_path, USERS_VAL_NEGA_FILE), allow_pickle=True)
                histories_val = np.load(os.path.join(self._save_path, HISTORIES_VAL_NEGA_FILE), allow_pickle=True)
                items_val = np.load(os.path.join(self._save_path, ITEMS_VAL_NEGA_FILE), allow_pickle=True)
                labels_val = np.load(os.path.join(self._save_path, LABEL_VAL_NEGA_FILE)).astype(np.float32)
            users_test = np.load(os.path.join(self._save_path, USERS_TEST_FILE), allow_pickle=True)
            histories_test = np.load(os.path.join(self._save_path, HISTORIES_TEST_FILE), allow_pickle=True)
            items_test = np.load(os.path.join(self._save_path, ITEMS_TEST_FILE), allow_pickle=True)
            labels_test = np.load(os.path.join(self._save_path, LABEL_TEST_FILE)).astype(np.float32)
            self.user_recall_test = np.load(os.path.join(self._save_path, USER_RECALL_TEST_FILE))
            self.histories_recall_test = np.load(os.path.join(self._save_path, HISTORIES_RECALL_TEST_FILE))
            self.item_recall_test = np.load(os.path.join(self._save_path, ITEM_RECALL_TEST_FILE))
            # infos = np.load(os.path.join(self._save_path, INFO_FILE), allow_pickle=True)
            with open(os.path.join(self._save_path, INFO_FILE), mode='rb') as f:
                infos = pickle.load(f)
            with open(os.path.join(self._save_path, USER_2_ITEM_TEST_DIC_FILE), mode='rb') as f:
                self._user_2_item_test_dic = pickle.load(f)
            with open(os.path.join(self._save_path, USER_2_ITEM_TRAIN_DIC_FILE), mode='rb') as f:
                self._user_2_item_train_dic = pickle.load(f)
            with open(os.path.join(self._save_path, TEST_ITEMS_FILE), mode='rb') as f:
                self._test_items = pickle.load(f)
            with open(os.path.join(self._save_path, ALL_ITEMS_FILE), mode='rb') as f:
                self._all_items = pickle.load(f)
            with open(os.path.join(self._save_path, TEST_USERS_FILE), mode='rb') as f:
                self._test_users = pickle.load(f)

            self._num_users, self._num_items = infos[0], infos[1]
            self._user_fea_fields, self._item_fea_fields = infos[2], infos[3]
            self._num_user_feas, self._num_item_feas = len(self._user_fea_fields), len(self._item_fea_fields)
            self._num_train_data = len(labels_train)
            self._num_val_data = len(labels_val)
            self._num_test_data = len(labels_test)
            self._num_data = self._num_train_data + self._num_val_data + self._num_test_data

            self._train_dataset = MyDataset(users_train, histories_train, items_train, labels_train)
            self._val_dataset = MyDataset(users_val, histories_val, items_val, labels_val)
            self._test_dataset = MyDataset(users_test, histories_test, items_test, labels_test)
        else:
            raise Exception(f"No preprocessed file!")
        if os.path.exists(os.path.join(self._save_path, PRETRAINED_EMBEDDINGS_DIR, BEST_PRETRAINED_EMBEDDINGS_FILE)) and self._use_pretrained_embeddings:
            self._pretrained_embeddings = torch.load(os.path.join(self._save_path, PRETRAINED_EMBEDDINGS_DIR, BEST_PRETRAINED_EMBEDDINGS_FILE))
        else:
            self._use_pretrained_embeddings = False
        self._logger.info(f"Num data {self._num_data}, "
                          f"Num train data {self._num_train_data}, "
                          f"Num val data {self._num_val_data}, "
                          f"Num test data {self._num_test_data}, "
                          f"Num users {self._num_users}, "
                          f"Num items {self._num_items}, "
                          f"Num test users {len(self._test_users)}, "
                          f"Num test items {len(self._test_items)}, "
                          f"Num user feas {self._num_user_feas}, "
                          f"Num item feas {self._num_item_feas}, "
                          f"User feas dim {self._user_fea_fields}, "
                          f"Item feas dim {self._item_fea_fields}")

    def _check_preprocess_file(self):
        return all([os.path.exists(os.path.join(self._save_path, file)) for file in FILES])

    def _split_dataset(self):
        print('split start')
        self._num_data = len(self._labels)
        self._num_train_data = int(self._num_data * self._train_ratio)
        self._num_val_data = int(self._num_data * self._val_ratio)
        self._num_test_data = self._num_data - self._num_train_data - self._num_val_data
        
        users_train, histories_train, items_train, labels_train = self._users[:self._num_train_data], self._histories[:self._num_train_data], self._items[:self._num_train_data], self._labels[:self._num_train_data]
        users_val, histories_val, items_val, labels_val = self._users[self._num_train_data:self._num_train_data + self._num_val_data], self._histories[self._num_train_data:self._num_train_data + self._num_val_data], self._items[self._num_train_data:self._num_train_data + self._num_val_data], self._labels[self._num_train_data:self._num_train_data + self._num_val_data]
        users_test, histories_test, items_test, labels_test = self._users[self._num_train_data + self._num_val_data:], self._histories[self._num_train_data + self._num_val_data:], self._items[self._num_train_data + self._num_val_data:], self._labels[self._num_train_data + self._num_val_data:]
        self._user_2_item_test_dic = collections.defaultdict(list)
        self._user_2_item_train_dic = collections.defaultdict(list)
        self._user_2_item_dic = collections.defaultdict(set)
        # user_2_index = collections.defaultdict(list)
        self._test_users = []
        self._test_items = []

        self._all_items = [[] for _ in range(self._num_items)]

        for idx, (u, i) in tqdm(enumerate(zip(self._users, self._items))):
            self._user_2_item_dic[u[0]].add(i[0])

        for i in tqdm(self._items):
            self._all_items[i[0]] = i
        self._all_items[0] = [0] * len(self._all_items[1])

        for u, i in tqdm(zip(users_train, items_train)):
            self._user_2_item_train_dic[u[0]].append(i[0])

        users_recall_test, histories_recall_test, items_recall_test = [], [], []
        test_users_set, test_items_set = set(), set()
        for u, h, i in tqdm(zip(users_test, histories_test, items_test)):
            self._user_2_item_test_dic[u[0]].append(i[0])
            if u[0] not in test_users_set:
                test_users_set.add(u[0])
                self._test_users.append(u[0])
                users_recall_test.append(u)
                histories_recall_test.append(h)
                
            if i[0] not in test_items_set:
                test_items_set.add(i[0])
                self._test_items.append(i[0])
                items_recall_test.append(i)

        # negative sample
        users_nega, histories_nega, items_nega, labels_nega = [], [], [], []
        existed = 0
        non_existed = 0

        if self._negative_sample > 0:
            all_idx = list(range(len(self._items)))
            for i in tqdm(range(len(users_train))):
                targets = random.sample(all_idx, self._negative_sample)
                cnt = 0
                for j in targets:
                    non_existed += 1
                    users_nega.append(users_train[i].copy())
                    histories_nega.append(histories_train[i].copy())
                    items_nega.append(self._items[j].copy())
                    labels_nega.append(0.0)
                    cnt += 1
                    if cnt >= self._negative_sample:
                        break
            for i in tqdm(range(len(users_val))):
                targets = random.sample(all_idx, self._negative_sample)
                cnt = 0
                for j in targets:
                    non_existed += 1
                    users_nega.append(users_val[i].copy())
                    histories_nega.append(histories_val[i].copy())
                    items_nega.append(self._items[j].copy())
                    labels_nega.append(0.0)
                    cnt += 1
                    if cnt >= self._negative_sample:
                        break

        users_nega.extend(users_train)
        histories_nega.extend(histories_train)
        items_nega.extend(items_train)
        labels_nega.extend(labels_train)
        users_nega.extend(users_val)
        histories_nega.extend(histories_val)
        items_nega.extend(items_val)
        labels_nega.extend(labels_val)
        train_num = int(len(users_nega) * (self._train_ratio / (self._train_ratio + self._val_ratio)))
        users_train_nega, histories_train_nega, items_train_nega, labels_train_nega = users_nega[:train_num], histories_nega[:train_num], items_nega[:train_num], labels_nega[:train_num]
        users_val_nega, histories_val_nega, items_val_nega, labels_val_nega = users_nega[train_num:], histories_nega[train_num:], items_nega[train_num:], labels_nega[train_num:]

        print(f'Count negative {existed}, {non_existed}')

        path = os.path.join(self._save_path, 'splits')
        try:
            shutil.rmtree(path)
        except Exception as e:
            pass
        os.makedirs(path)

        files = {USERS_TRAIN_FILE: users_train, ITEMS_TRAIN_FILE: items_train, LABEL_TRAIN_FILE: labels_train, HISTORIES_TRAIN_FILE: histories_train,
                 USERS_TRAIN_NEGA_FILE: users_train_nega, ITEMS_TRAIN_NEGA_FILE: items_train_nega, LABEL_TRAIN_NEGA_FILE: labels_train_nega, HISTORIES_TRAIN_NEGA_FILE: histories_train_nega,
                 USERS_VAL_FILE: users_val, ITEMS_VAL_FILE: items_val, LABEL_VAL_FILE: labels_val, HISTORIES_VAL_FILE: histories_val,
                 USERS_VAL_NEGA_FILE: users_val_nega, ITEMS_VAL_NEGA_FILE: items_val_nega, LABEL_VAL_NEGA_FILE: labels_val_nega, HISTORIES_VAL_NEGA_FILE: histories_val_nega,
                 USERS_TEST_FILE: users_test, ITEMS_TEST_FILE: items_test, LABEL_TEST_FILE: labels_test, HISTORIES_TEST_FILE: histories_test,
                 USER_RECALL_TEST_FILE: users_recall_test, HISTORIES_RECALL_TEST_FILE: histories_recall_test, ITEM_RECALL_TEST_FILE: items_recall_test}
        for file, data in files.items():
            np.save(os.path.join(self._save_path, file), data)
        with open(os.path.join(self._save_path, INFO_FILE), 'wb') as f:
            pickle.dump([self._num_users, self._num_items, self._user_fea_fields, self._item_fea_fields], f)
        with open(os.path.join(self._save_path, USER_2_ITEM_TEST_DIC_FILE), 'wb') as f:
            pickle.dump(self._user_2_item_test_dic, f)
        with open(os.path.join(self._save_path, USER_2_ITEM_TRAIN_DIC_FILE), 'wb') as f:
            pickle.dump(self._user_2_item_train_dic, f)
        # with open(os.path.join(self._save_path, USER_2_ITEM_DIC_FILE), 'wb') as f:
        #     pickle.dump(self._user_2_item_dic, f)
        with open(os.path.join(self._save_path, TEST_USERS_FILE), 'wb') as f:
            pickle.dump(self._test_users, f)
        with open(os.path.join(self._save_path, TEST_ITEMS_FILE), 'wb') as f:
            pickle.dump(self._test_items, f)
        with open(os.path.join(self._save_path, ALL_ITEMS_FILE), 'wb') as f:
            pickle.dump(self._all_items, f)

        print('split end')

    def _transform(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def num_data(self):
        return self._num_data

    @property
    def num_train_data(self):
        return self._num_train_data

    @property
    def num_val_data(self):
        return self._num_val_data

    @property
    def num_test_data(self):
        return self._num_test_data

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def user_fea_fields(self):
        return self._user_fea_fields

    @property
    def item_fea_fields(self):
        return self._item_fea_fields

    @property
    def save_path(self):
        return self._save_path

    @property
    def pretrained_embeddings(self):
        return self._pretrained_embeddings
    
    @property
    def user_2_item_test_dic(self):
        return self._user_2_item_test_dic

    @property
    def user_2_item_train_dic(self):
        return self._user_2_item_train_dic

    @property
    def test_users(self):
        return self._test_users

    @property
    def test_items(self):
        return self._test_items

    @property
    def all_items(self):
        return self._all_items

    @property
    def user_fea_emb_dim(self):
        return self._user_fea_emb_dim

    @property
    def item_fea_emb_dim(self):
        return self._item_fea_emb_dim
