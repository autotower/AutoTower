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


class YelpDataset(Dataset):
    def __init__(self, data_path, negative_sample=0, repreprocess=False, use_pretrained_embeddings=False, logger=None):
        super(YelpDataset, self).__init__(
            name='yelp',
            data_path=data_path,
            num_user_feas=6,
            num_item_feas=6,
            repreprocess=repreprocess,
            use_pretrained_embeddings=use_pretrained_embeddings,
            with_head=False,
            negative_sample=negative_sample,
            logger=logger)

        self.seq_len = 40
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
        super(YelpDataset, self)._read()

    def _transform(self, *args, **kwargs):
        """Yalp Dataset
        
        field:
            user_id, user_review_count, yelp_since, usefule, funny, cool,
            business_id, city, state, stars, b_review_count, is_open

        Data Info:
            0 stars
            1 user_id
            2 business_id
            ------ review -------
            3 useful
            4 funny
            5 cool
            6 date
            ------ user --------
            7 review_count
            8 yelping_since
            9 useful
            10 funny
            11 cool
            ------- business ------
            12 city
            13 state
            14 stars
            15 review_count
            16 is_open
        """
        
        data = pd.read_csv(self._data_path)
        lbe = LabelEncoder()
        
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
        data.sort_values(by='date', ignore_index=True, inplace=True)
        
        data['user_id'] = lbe.fit_transform(data['user_id']) + 1
        data['business_id'] = lbe.fit_transform(data['business_id']) + 1
        
        data['city'] = lbe.fit_transform(data['city'])
        data['state'] = lbe.fit_transform(data['state'])
        
        self._num_users = int(data['user_id'].max()) + 1
        self._user_fea_fields = [self._num_users, int(data['review_count_x'].max()) + 1,
                                 int(data['yelping_since'].max()) + 1, int(data['useful_x'].max()) + 1,
                                 int(data['funny_x'].max()) + 1, int(data['cool_x'].max()) + 1]
        self._num_items = int(data['business_id'].max()) + 1
        self._item_fea_fields = [self._num_items, int(data['city'].max()) + 1,
                                 int(data['state'].max()) + 1, int(data['stars_y'].max()) + 1,
                                 int(data['review_count_y'].max()) + 1, int(data['is_open'].max()) + 1]
        
        users_train, users_test, items_train, items_test, histories_train, histories_test = [], [], [], [], [], []
        for reviewerID, feats in tqdm(data.groupby('user_id')):
            ninters = len(feats)
            ntrain = int(ninters * (self._train_ratio + self._val_ratio))
            user_feats = feats[['user_id', 'review_count_x', 'yelping_since', 'useful_x', 'funny_x', 'cool_x']].values.tolist()
            item_feats = feats[['business_id', 'city', 'state', 'stars_y', 'review_count_y', 'is_open']].values.tolist()
            
            business_list = feats['business_id'].tolist()
            city_list = feats['city'].tolist()
            state_list = feats['state'].tolist()
            stars_list = feats['stars_y'].tolist()
            review_count_list = feats['review_count_y'].tolist()
            is_open_list = feats['is_open'].tolist()
            
            histories = []
            for i in range(len(business_list)):
                if i >= self.seq_len:
                    hist = [business_list[i - self.seq_len:i], city_list[i - self.seq_len:i],
                            state_list[i - self.seq_len: i], stars_list[i - self.seq_len: i],
                            review_count_list[i - self.seq_len: i], is_open_list[i - self.seq_len: i]]
                else:
                    hist = [[0] * (self.seq_len - i) + business_list[:i],
                            [0] * (self.seq_len - i) + city_list[:i],
                            [0] * (self.seq_len - i) + state_list[:i],
                            [0] * (self.seq_len - i) + stars_list[:i],
                            [0] * (self.seq_len - i) + review_count_list[:i],
                            [0] * (self.seq_len - i) + is_open_list[:i]
                            ]
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
        
        # users = data[['user_id', 'age', 'gender', 'occupation']].values.tolist()
        # items = data[['movie_id', 'year']].values
        # items = np.concatenate((items, np.asarray(data['generes'].tolist())), axis=-1).tolist()

        labels = [1.0] * len(users)
        # users, items, labels = shuffle(users, items, labels)

        return users, histories, items, labels


if __name__ == '__main__':
    save_dir = os.path.join(PROJECT_PATH, 'logs', 'datasets')
    check_directory(save_dir, force_removed=False)
    save_file = os.path.join(save_dir, 'log_dataset_yelp.txt')
    check_file(save_file, force_removed=True)
    LOGGER = get_logger(AUTOTOWER, save_file)
    data_path = '/data/yelp/yelp.txt'
    dataset = YelpDataset(data_path, negative_sample=0, repreprocess=True, logger=LOGGER)
