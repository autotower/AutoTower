#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))
import time
import collections
import re
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from codes.datasets.dataset import Dataset
from codes.utils.utils import PROJECT_PATH, AUTOTOWER, check_file, get_logger, check_directory


class AmazonBookDataset(Dataset):
    def __init__(self, data_path, negative_sample=0, repreprocess=False, use_pretrained_embeddings=False, logger=None):
        super(AmazonBookDataset, self).__init__(
            name='amazon_book',
            data_path=data_path,
            num_user_feas=1,
            num_item_feas=2,
            repreprocess=repreprocess,
            use_pretrained_embeddings=use_pretrained_embeddings,
            with_head=False,
            negative_sample=negative_sample,
            logger=logger)

        self.seq_len = 50
        self._read()

    def _read(self):
        if not self._check_preprocess_file() or self._repreprocess:
            self._logger.info("reprocess dataset")
            self._users, self._histories, self._items, self._labels = self._transform()
            # self._num_users = len(np.unique(self._users))
            # self._num_items = len(np.unique(self._items))
            # self._logger.info(f"users info {self._num_users} {max(self._users)} {min(self._users)}")
            # self._logger.info(f"items info {self._num_items} {max(self._items)} {min(self._items)}")
            self._split_dataset()
        super(AmazonBookDataset, self)._read()

    def _transform(self, *args, **kwargs):
        """AmazonBook Dataset
        
        field:
            reviewerID, asin, price, time
        """
        
        data = pd.read_csv(self._data_path)
        lbe = LabelEncoder()
        
        data.sort_values(by='time', ignore_index=True, inplace=True)
        
        data['reviewerID'] = lbe.fit_transform(data['reviewerID']) + 1
        data['asin'] = lbe.fit_transform(data['asin']) + 1
        
        self._num_users = int(data['reviewerID'].max()) + 1
        self._user_fea_fields = [self._num_users]
        self._num_items = int(data['asin'].max()) + 1
        self._item_fea_fields = [self._num_items, int(data['price'].max()) + 1]
        
        users_train, users_test, items_train, items_test, histories_train, histories_test = [], [], [], [], [], []
        for reviewerID, feats in tqdm(data.groupby('reviewerID')):
            ninters = len(feats)
            ntrain = int(ninters * (self._train_ratio + self._val_ratio))
            user_feats = feats[['reviewerID']].values.tolist()
            item_feats = feats[['asin', 'price']].values.tolist()
            
            item_list = feats['asin'].tolist()
            price_list = feats['price'].tolist()
            
            histories = []
            for i in range(len(item_list)):
                if i >= self.seq_len:
                    hist = [item_list[i - self.seq_len:i], price_list[i - self.seq_len:i]]
                else:
                    hist = [[0] * (self.seq_len - i) + item_list[:i],
                            [0] * (self.seq_len - i) + price_list[:i]]
                histories.append(hist)
            
            users_train.extend(user_feats[:ntrain])
            users_test.extend(user_feats[ntrain:])
            items_train.extend(item_feats[:ntrain])
            items_test.extend(item_feats[ntrain:])
            histories_train.extend(histories[:ntrain])
            histories_test.extend(histories[ntrain:])
            
        users_train, items_train, histories_train = shuffle(users_train, items_train, histories_train)
        users_test, items_test, histories_test = shuffle(users_test, items_test, histories_test)
        
        users_train.extend(users_test)
        items_train.extend(items_test)
        histories_train.extend(histories_test)
        
        users, histories, items = users_train, histories_train, items_train
        
        labels = [1.0] * len(users)

        return users, histories, items, labels


if __name__ == '__main__':
    save_dir = os.path.join(PROJECT_PATH, 'logs', 'datasets')
    check_directory(save_dir, force_removed=False)
    save_file = os.path.join(save_dir, 'log_dataset_AmazonBook.txt')
    check_file(save_file, force_removed=True)
    LOGGER = get_logger(AUTOTOWER, save_file)
    data_path = '/data/amazon/amazon_book/amazon_book.txt'
    dataset = AmazonBookDataset(data_path, negative_sample=0, repreprocess=True, logger=LOGGER)
