import collections

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import multiprocessing

import heapq

cores = multiprocessing.cpu_count() // 6

from codes.utils.utils import profile
from codes.datasets.dataset import Dataset
from codes.utils.ksmallest import ksmallest


def supernet_evaluate(supernet, test_queue, use_gpu, cal_hr=False, topk=None, arch=None):
    supernet.eval()
    loss_type = supernet.loss_type

    xs, ys = [], []
    losses = []
    auc_list = []
    hr_topk = collections.defaultdict(list)
    with torch.no_grad():
        for users, histories, items, labels in test_queue:
            if use_gpu:
                users, histories, items, labels = users.cuda(), histories.cuda(), items.cuda(), labels.cuda()
            if not arch:
                arch = supernet.get_uniform_sample_arch()
            preds, _ = supernet(users, histories, items, arch)

            if loss_type == 'bce':
                xs.append(preds.flatten())
                ys.append(labels)
            elif loss_type == 'batch_softmax':
                _, loss = get_batch_softmax_loss(preds, labels)
                auc = compute_auc(labels, preds)
                losses.append(loss.cpu().detach().item())
                auc_list.append(auc)
                if cal_hr:
                    for k in topk:
                        hr = compute_hit_ratio(labels, preds, k=k)
                        hr_topk[k].append(hr.cpu().detach().item())
            else:
                raise Exception(f'No such loss type: {loss_type}')

    if loss_type == 'bce':
        preds, labels = torch.cat(xs), torch.cat(ys)
        loss = torch.nn.BCEWithLogitsLoss()(preds, labels)
        preds = torch.sigmoid(preds)
        try:
            auc = roc_auc_score(labels.cpu(), preds.cpu())
        except ValueError as e:
            print(e)
            auc = 0
    elif loss_type == 'batch_softmax':
        hr_topk = {k: np.mean(v) for k, v in hr_topk.items()}
        loss = np.mean(losses)
        auc = np.mean(auc_list)
    else:
        raise Exception(f'No such loss type: {loss_type}')

    return loss, auc, hr_topk


def evaluate(model, test_queue, use_gpu, cal_flops=True, cal_hr=True, topk=None):
    model.eval()
    flops, params = None, None
    loss_type = model.loss_type

    xs, ys = [], []
    losses = []
    auc_list = []
    hr_topk = collections.defaultdict(list)
    with torch.no_grad():
        for users, histories, items, labels in test_queue:
            if use_gpu:
                users, histories, items, labels = users.cuda(), histories.cuda(), items.cuda(), labels.cuda()
            preds, _ = model(users, histories, items)

            if loss_type == 'bce':
                xs.append(preds.flatten())
                ys.append(labels)
            elif loss_type == 'sampled_softmax':
                xs.append(preds)
                ys.append(labels)
            elif loss_type == 'batch_softmax':
                _, loss = get_batch_softmax_loss(preds, labels)
                auc = compute_auc(labels, preds)
                losses.append(loss.cpu().detach().item())
                auc_list.append(auc)
                if cal_hr:
                    for k in topk:
                        hr = compute_hit_ratio(labels, preds, k=k)
                        hr_topk[k].append(hr.cpu().detach().item())
            else:
                raise Exception(f'No such loss type: {loss_type}')

            if flops is None and cal_flops:
                flops, params = profile(model, users, histories, items)
                flops = flops #/ (1000 ** 3)
                params = params # / (1000 ** 2)

    if loss_type == 'bce':
        preds, labels = torch.cat(xs), torch.cat(ys)
        loss = torch.nn.BCEWithLogitsLoss()(preds, labels)
        preds = torch.sigmoid(preds)
        try:
            auc = roc_auc_score(labels.cpu(), preds.cpu())
        except ValueError as e:
            print(e)
            auc = 0
    elif loss_type == 'sampled_softmax':
        preds, labels = torch.cat(xs), torch.cat(ys)
        loss = get_sampled_softmax_loss(preds, labels)
        auc = 0
    elif loss_type == 'batch_softmax':
        hr_topk = {k: np.mean(v) for k, v in hr_topk.items()}
        loss = np.mean(losses)
        auc = np.mean(auc_list)
    else:
        raise Exception(f'No such loss type: {loss_type}')

    return loss, auc, hr_topk, params, flops


def get_sampled_softmax_loss(out, label):
    logits = F.softmax(out)
    loss = -torch.sum(torch.log(logits[:, 0])) / out.shape[0]

    return loss

def get_batch_softmax_loss(out, label, sample_rate=1.0, mask=None):
    logit = out - torch.log(torch.tensor(sample_rate))
    pred = torch.softmax(logit, dim=1)
    loss = -torch.sum(torch.log(torch.diag(pred))) / label.shape[0]
    if mask is not None:
        zeros = torch.zeros_like(loss, dtype=loss.dtype)
        loss = torch.where(mask, loss, zeros)
    # print(torch.mean(pred).cpu().item(), torch.max(pred).cpu().item(), loss)
    # loss = torch.mean(loss)
    return pred, loss

def get_corrected_batch_softmax_loss(out, label, A: dict, B: dict, t, items,
                                     pred_init, alpha, temperature=0.05, sample_rate=1.0, mask=None):
    """
    Compute corrected batch softmax loss by frequency estimation
    
    Hyper-parameters:
        pred_init: number of items / batch size
        alpha: learning rate of correclation
        temperature: softmax temperature

    Args:
        out: predicted result
        label:
        A: maintain appeared items
        B: maintain appeared items
        t: step num
        items: items in current batch
        alpha: learning rate
    """
    p = [] # correction
    for item in items:
        item = item[0].item()
        B[item] = (1 - alpha) * B.get(item, pred_init) + alpha * (t - A.get(item, 0))
        A[item] = t
        p.append(1 / B[item])
    
    logits = out / temperature - torch.log(torch.tensor(p, device=out.device))

    preds = torch.softmax(logits, dim=1)
    
    loss = -torch.sum(torch.log(torch.diag(preds))) / label.shape[0]
    
    return preds, loss


def compute_auc(label, pred):
    return 0


def compute_hit_ratio(label, pred, k, use_gpu=True):
    soft_label = torch.eye(label.shape[0]).to(pred.device)
    soft_label = torch.ge(soft_label, 0.5)
    pos_score = torch.sum(torch.mul(soft_label, pred), dim=1, keepdim=True).repeat([1, pred.shape[1]])
    pos_rank = torch.sum(torch.lt(pos_score + 1e-6, pred), dim=1, dtype=torch.int32) + 1
    hit_ratio = torch.mean(torch.le(pos_rank, k).float())

    return hit_ratio


def recall_at_k(r, k, all_inter_num):
    """Compute recall rate of one user
    """
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_inter_num


def dcg_at_k(r, k, method=1):
    """Discounted cumulative gain (dcg)
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('dcg method must be 0 or 1')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Normalized discounted cumulative gain (ndcg).
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def mrr_at_k(r, k):
    """Compute MRR@K of one user
    """
    r = np.asfarray(r)[:k]
    if np.sum(r) > 0:
        return 1 / (np.argmax(r) + 1)
    else:
        return 0.


def ranklist_by_heapq(user_inter_items, test_items, rating, topks):
    """
    Compute whether the top max K items in test items.

    Args:
        user_inter_items: user interacted items in the test set
        test_items: items used to caculate hit ratio
    """
    # item_score = {}
    # for i in test_items:
    #     item_score[i] = rating[i]
    
    # test_items = np.asarray(test_items, dtype=np.int32)
    item_score = rating[test_items]
    
    K_max = max(topks)
    # K_max_item_score = heapq.nsmallest(K_max, item_score, key=item_score.get)
    K_max_item_score, scores = ksmallest(K_max, test_items, item_score)    
    K_max_item_score, scores = zip(*sorted(zip(K_max_item_score, scores), key=lambda x: x[1]))
    
    r = []
    for item in K_max_item_score:
        if item in user_inter_items:
            r.append(1)
        else:
            r.append(0)
    return r


def eval_one_user(x):
    user = x[0]
    rating = x[1]

    # user's interactive items in the training set
    training_items = _dataset.user_2_item_train_dic[user]
    training_items.append(0)
    
    # user's interactive items in the test set
    user_inter_items = _dataset.user_2_item_test_dic[user]

    # all_items = set(range(_dataset.num_items))
    # test_items = list(all_items - set(training_items))  # remove items in training set
    
    all_items = np.arange(_dataset.num_items, dtype=np.int32)
    mask = np.ones_like(all_items, bool)
    mask[training_items] = False
    test_items = all_items[mask]
    # test_items = np.delete(all_items, training_items)
    
    r = ranklist_by_heapq(user_inter_items, test_items, rating, _topks)
    return [recall_at_k(r, k, len(user_inter_items)) for k in _topks], [ndcg_at_k(r, k) for k in _topks] , [mrr_at_k(r, k) for k in _topks]

@torch.no_grad()
def get_recall_ratio(model, dataset: Dataset, topks, use_gpu=True, max_block=512):

    global _dataset
    global _topks
    _dataset = dataset
    _topks = topks

    pool = multiprocessing.Pool(cores)

    # u2i_dic = dataset.user_2_item_test_dic
    device = 'cuda' if use_gpu else 'cpu'
    users_feats = torch.tensor(dataset.user_recall_test, device=device)
    items_feats = torch.tensor(dataset.all_items, device=device)
    users = dataset.test_users

    # if use_gpu:
    #     users_feats, items_feats = users_feats.cuda(), items_feats.cuda()

    user_embs, item_embs = model(users_feats, items_feats, only_tower_output=True)
    
    del users_feats, items_feats

    n_k = len(topks)
    user_rate = []
    results = []
    for i, (user, user_emb) in enumerate(zip(users, user_embs)):
        user_emb = user_emb.reshape([1, -1]).repeat([len(items_feats), 1])
        dists = torch.sum((user_emb - item_embs).pow(2), dim=1).tolist()
        user_rate.append(dists)
        if (i + 1) % max_block == 0 or i == len(users) - 1:
            results.extend(pool.map(eval_one_user,
                                    zip(users[(i // max_block) * max_block: i + 1],
                                        user_rate)))
            user_rate.clear()
            
    recall_rate = collections.defaultdict(float)

    # results = pool.map(eval_one_user, zip(users, user_rate))
    recall_results, _, _ = zip(*results)
    recall_results = np.asarray(recall_results).mean(axis=0)

    for i in range(n_k):
        recall_rate[topks[i]] = recall_results[i]

    return recall_rate


@torch.no_grad()
def get_recall_ncdg_mrr(model, dataset, topks, use_gpu=True, max_block=512):
    
    import time
    btime = time.time()

    global _dataset
    global _topks
    _dataset = dataset
    _topks = topks

    pool = multiprocessing.Pool(cores)

    # u2i_dic = dataset.user_2_item_test_dic
    device = 'cuda' if use_gpu else 'cpu'
    users_feats = torch.tensor(dataset.user_recall_test, device=device)
    histories = torch.tensor(dataset.histories_recall_test, device=device)
    items_feats = torch.tensor(dataset.all_items, device=device)
    users = dataset.test_users

    # if use_gpu:
    #     users_feats, items_feats = users_feats.cuda(), items_feats.cuda()

    user_embs, item_embs = model(users_feats, histories, items_feats, only_tower_output=True)
    
    del users_feats, items_feats
    
    n_k = len(topks)
    user_rate = []
    results = []
    for i, (user, user_emb) in enumerate(zip(users, user_embs)):
        if len(user_embs.shape) == 3:
            user_emb = user_emb.reshape([1, user_embs.shape[1], -1]).repeat([dataset.num_items, 1, 1])
            # dists = torch.sum((user_emb - item_embs).pow(2), dim=1).tolist()
            dists = torch.sum((user_emb - item_embs.unsqueeze(1)).pow(2), dim=2)
            dists, _ = torch.min(dists, dim=1)
            dists = dists.cpu().numpy()
        else:
            user_emb = user_emb.reshape([1, -1]).repeat([dataset.num_items, 1])
            # dists = torch.sum((user_emb - item_embs).pow(2), dim=1).tolist()
            dists = torch.sum((user_emb - item_embs).pow(2), dim=1).cpu().numpy()
        user_rate.append(dists)
        
        if (i + 1) % max_block == 0 or i == len(users) - 1:
            results.extend(pool.map(eval_one_user,
                                    zip(users[(i // max_block) * max_block: i + 1],
                                        user_rate)))

            # result = get_metrics(users[(i // max_block) * max_block: i + 1], user_rate,
            #             dict(dataset.user_2_item_train_dic), dict(dataset.user_2_item_test_dic),
            #             dataset.num_items,topks, 4)
            # print(result)
            user_rate.clear()

    recall_rate = collections.defaultdict(float)
    mrr = collections.defaultdict(float)
    ncdg = collections.defaultdict(float)

    # results = pool.map(eval_one_user, zip(users, user_rate))
    recall_results, ncdg_results, mrr_results = zip(*results)
    recall_results = np.asarray(recall_results).mean(axis=0)
    ncdg_results = np.asarray(ncdg_results).mean(axis=0)
    mrr_results = np.asarray(mrr_results).mean(axis=0)

    for i in range(n_k):
        recall_rate[topks[i]] = recall_results[i]
        ncdg[topks[i]] = ncdg_results[i]
        mrr[topks[i]] = mrr_results[i]
        
    print(f'compute recall time:{time.time() - btime}')

    return recall_rate, ncdg, mrr

@torch.no_grad()
def get_recall_ncdg_mrr_with_score(model, dataset, topks, use_gpu=True, max_block=512):
    """Compute evaluation metrics direcly by score.
    
    For FM, NCF etc.
    """
    global _dataset
    global _topks
    _dataset = dataset
    _topks = topks

    pool = multiprocessing.Pool(cores)

    # u2i_dic = dataset.user_2_item_test_dic
    device = 'cuda' if use_gpu else 'cpu'
    users_feats = torch.tensor(dataset.user_recall_test, device=torch.device(device))
    histories = torch.tensor(dataset.histories_recall_test, device=torch.device(device))
    items_feats = torch.tensor(dataset.all_items, device=torch.device(device))
    users = dataset.test_users

    # if use_gpu:
    #     users_feats, items_feats = users_feats.cuda(), items_feats.cuda()

    n_k = len(topks)
    user_rate = []
    results = []
    for i, (user, user_feat, history) in enumerate(zip(users, users_feats, histories)):
        user_feat = user_feat.repeat([items_feats.shape[0], 1])
        history = history.repeat([items_feats.shape[0], 1])
        scors, _ = model(user_feat, history, items_feats)
        # dists = (-scors.flatten()).tolist()
        dists = (-scors.flatten()).cpu().numpy()
        
        user_rate.append(dists)
        if (i + 1) % max_block == 0 or i == len(users) - 1:
            results.extend(pool.map(eval_one_user,
                                    zip(users[(i // max_block) * max_block: i + 1],
                                        user_rate)))
            user_rate.clear()

    recall_rate = collections.defaultdict(float)
    mrr = collections.defaultdict(float)
    ncdg = collections.defaultdict(float)

    # results = pool.map(eval_one_user, zip(users, user_rate))
    recall_results, ncdg_results, mrr_results = zip(*results)
    recall_results = np.asarray(recall_results).mean(axis=0)
    ncdg_results = np.asarray(ncdg_results).mean(axis=0)
    mrr_results = np.asarray(mrr_results).mean(axis=0)

    for i in range(n_k):
        recall_rate[topks[i]] = recall_results[i]
        ncdg[topks[i]] = ncdg_results[i]
        mrr[topks[i]] = mrr_results[i]

    return recall_rate, ncdg, mrr


def compare_recall_ratio_list(l1, l2, keys=None):
    if l1 is None:
        return True
    if l2 is None:
        return False
    l1_cnt, l2_cnt = 0, 0
    l1_sum, l2_sum = 0, 0

    if keys is None:
        keys = l1.keys()
    for k in keys:
        l1_sum += l1[k]
        l2_sum += l2[k]
        if l1[k] >= l2[k]:
            l1_cnt += 1
        else:
            l2_cnt += 1
    if l1_cnt == l2_cnt:
        if l1_sum >= l2_sum:
            l1_cnt += 1
        else:
            l2_cnt += 1

    return l1_cnt < l2_cnt
