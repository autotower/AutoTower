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
from codes.utils.utils import PROJECT_PATH, AUTOTOWER, get_logger, check_directory


class Movielens100kDataset(Dataset):
    def __init__(self, data_path, negative_sample=0, repreprocess=False, use_pretrained_embeddings=False, logger=None):
        super(Movielens100kDataset, self).__init__(
            name='ml100k',
            data_path=data_path,
            num_user_feas=4,
            num_item_feas=3,
            repreprocess=repreprocess,
            use_pretrained_embeddings=use_pretrained_embeddings,
            with_head=False,
            negative_sample=negative_sample,
            logger=logger)

        self.seq_len = 20
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
        super(Movielens100kDataset, self)._read()

    def _transform(self, *args, **kwargs):
        """
        field:
            rating user_id movie_id age gender occupation year generes
        """
        
        data = pd.read_csv(self._data_path)
        lbe = LabelEncoder()
        
        data.sort_values(by='timestamp', ignore_index=True, inplace=True)
        
        def year_map(x):
            if x == 'unknown':
                return 0
            year = int(re.match(r".*\((\d+)\)", x).group(1))
            if year is None:
                return 0
            elif year < 1900:
                return 1
            else:
                return (year - 1900) // 10
            
        def age_map(age):
            if age is None or age < 1:
                return 0
            elif 1 <= age <= 7:
                return 1
            elif 8 <= age <= 16:
                return 2
            elif 17 <= age <= 29:
                return 3
            elif 30 <= age <= 39:
                return 4
            elif 40 <= age <= 49:
                return 5
            elif 50 <= age <= 59:
                return 6
            else:
                return 7

        generes = data['generes'].map(lambda x: x.strip().split('|'))
        tokens = generes.agg(np.concatenate)
        tokens = lbe.fit_transform(tokens)
        ngenere = max(tokens) + 1
        split_point = np.cumsum(generes.agg(len))[:-1]
        generes = np.split(tokens, split_point)

        data['generes'] = generes
        data['generes_single'] = data['generes'].map(lambda x: x[0])

        data['generes'] = data['generes'].map(lambda x: np.concatenate(([len(x)], x, (ngenere - len(x)) * [ngenere])))

        data['year'] = data['title'].apply(year_map)
        data['user_id'] = lbe.fit_transform(data['user_id']) + 1
        data['movie_id'] = lbe.fit_transform(data['movie_id']) + 1
        data['gender'] = lbe.fit_transform(data['gender'])
        data['occupation'] = lbe.fit_transform(data['occupation'])
        data['age'] = data['age'].map(age_map)
        
        self._num_users = int(data['user_id'].max()) + 1
        self._user_fea_fields = [int(data['user_id'].max()) + 1, int(data['age'].max()) + 1, int(data['gender'].max()) + 1,
                                 int(data['occupation'].max()) + 1]
        self._num_items = int(data['movie_id'].max()) + 1
        self._item_fea_fields = [int(data['movie_id'].max()) + 1, int(data['year'].max()) + 1, ngenere]
        
        users_train, users_test, items_train, items_test, histories_train, histories_test = [], [], [], [], [], []
        for reviewerID, feats in tqdm(data.groupby('user_id')):
            ninters = len(feats)
            ntrain = int(ninters * (self._train_ratio + self._val_ratio))
            user_feats = feats[['user_id', 'age', 'gender', 'occupation']].values.tolist()
            item_feats = feats[['movie_id', 'year']].values
            item_feats = np.concatenate((item_feats, np.asarray(feats['generes'].tolist())), axis=-1).tolist()
            
            movie_list = feats['movie_id'].tolist()
            year_list = feats['year'].tolist()
            genere_list = feats['generes_single'].tolist()
            histories = []
            for i in range(len(movie_list)):
                if i >= self.seq_len:
                    hist = [movie_list[i - self.seq_len:i], year_list[i - self.seq_len:i],
                            genere_list[i - self.seq_len: i]]
                else:
                    hist = [[0] * (self.seq_len - i) + movie_list[:i],
                            [0] * (self.seq_len - i) + year_list[:i],
                            [0] * (self.seq_len - i) + genere_list[:i]]
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
    check_directory(save_dir, force_removed=True)
    save_file = os.path.join(save_dir, 'log_dataset_ml100k.txt')
    LOGGER = get_logger(AUTOTOWER, save_file)
    data_path = '/data/ml-100k/ml100k.txt'
    dataset = Movielens100kDataset(data_path, negative_sample=0, repreprocess=True, logger=LOGGER)
