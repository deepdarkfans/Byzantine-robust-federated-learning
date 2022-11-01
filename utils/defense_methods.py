from audioop import avg
import copy
import math
from functools import reduce

import numpy as np
import sklearn.metrics.pairwise as smp
import torch
from loguru import logger
from sklearn.decomposition import PCA
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances,euclidean_distances
from utils.agg_methods import average_weights, average_weights_rep, average_weights_edge

eps = np.finfo(float).eps


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def _compute_scores(distances, i, n, f):
    """Compute scores for node i.
    Args:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
        i, j starts with 0.
        i {int} -- index of worker, starting from 0.
        n {int} -- total number of workers
        f {int} -- Total number of Byzantine workers.
    Returns:
        float -- krum distance score of i.
    """
    s = [distances[j][i] ** 2 for j in range(i)] + [
        distances[i][j] ** 2 for j in range(i + 1, n)
    ]
    _s = sorted(s)[: n - f - 2]
    return sum(_s)


def _multi_krum(distances, n, f, m):
    """Multi_Krum algorithm.
    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
         i, j starts with 0.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.
    Returns:
        list -- A list indices of worker indices for aggregation. length <= m
    """
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if m < 1 or m > n:
        raise ValueError(
            "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                m, n
            )
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            if distances[i][j] < 0:
                raise ValueError(
                    "The distance between node {} and {} should be non-negative: "
                    "Got {}.".format(i, j, distances[i][j])
                )

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return list(map(lambda x: x[0], sorted_scores))[:m]


def Krum(updates, f, net_glob,multi=False ):
    n = len(updates)
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]
    updates_ = torch.empty([n, len(updates[0])])
    for i in range(n):
      updates_[i] = updates[i]
    k = n - f - 2
    # collection distance, distance from points to points
    cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k , largest=False)
    dist = dist.sum(1)
    idxs = dist.argsort()
    if multi:
      return idxs[:k]
    else:
      return idxs[0]


def Ours(updates, f, multi=False):
    n = len(updates)

    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]
    #fc3_vector=[torch.nn.utils.parameters_to_vector(update.fc3.bias) for update in updates]
    
    updates_ = torch.empty([n, len(updates[0])])
    for i in range(n):
        updates_[i] = updates[i]
    k = n - f
    # collection distance, distance from points to points
    cdist = torch.cosine_similarity(updates_.unsqueeze(1), updates_.unsqueeze(0), dim=-1)
    # cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k, largest=True)
    dist = dist.sum(1)
    idxs = dist.argsort(descending=True)
    if multi:
        return idxs[:k]
    else:
        return idxs[0]

def ClusterAVG(updates,net_glob,idxs_users):
    local_score=ClusterLocalVector(updates,net_glob)
    global_score=ClusterGlobelVector(updates,net_glob)
    # global_score = np.zeros(len(idxs_users))
    # global_score[idxs]=1
    if sum(global_score+local_score)!=0:
        final_score=np.zeros(len(idxs_users))
        avg_score=(local_score+global_score)/2
        for ind,label in enumerate(avg_score):
            if label>0.5:
                final_score[ind]=label
    # else:
    #     final_score=np.zeros(len(idxs_users))
    #     avg_score=(local_score+global_score)+1
    #     for ind,label in enumerate(avg_score):
    #         if label>=0.5:
    #             final_score[ind]=label
    
    return final_score

    
def ClusterLocalVector(updates,net_glob):
    fc3_vector=[torch.nn.utils.parameters_to_vector(update.fc3.bias) for update in updates]
    net_glob_vector=[torch.nn.utils.parameters_to_vector(net_glob.fc3.bias)]
    n = len(updates)
    #updates_ = torch.empty([n, len(updates[0])])
    # for i in range(n):
    #     updates_[i] = updates[i]
    fc3_vector_=torch.empty([n, len(updates[0].fc3.bias)])
    for i in range(n): 
        fc3_vector_[i]=fc3_vector[i]-net_glob_vector[0]
    #pca = PCA(n_components='mle',copy=False)
    #pca_fc3=pca.fit_transform(fc3_vector_)
    #cos_dist=cosine_distances(fc3_vector_.detach().numpy())
    #fc3_vector_norm=StandardScaler().fit_transform(fc3_vector_.detach().numpy())
    #pca = PCA(n_components=10,copy=False)
    #pca_fc3=pca.fit_transform(fc3_vector_norm)
    #cluterss=hdbscan.HDBSCAN(min_cluster_size=5,min_samples=1,metric='l2',prediction_data=True)
    cluterss=hdbscan.HDBSCAN(min_cluster_size=2,min_samples=1,metric='l2',prediction_data=True)
    #cluterss.fit(fc3_vector_.detach().numpy())
    cluterss.fit(fc3_vector_.detach().numpy())
    cluterss_labels=cluterss.labels_
    #cluterss_probabilities=cluterss.probabilities_
    soft_clusters = hdbscan.all_points_membership_vectors(cluterss)
    #score=np.zeros((len(np.unique(cluterss.labels_))-1,len(updates)))
    local_score=np.zeros(len(updates))
    for ind,label in enumerate(cluterss_labels):
          if label != -1:
            #score[label][ind]=soft_clusters[ind][label]
            local_score[ind]=soft_clusters[ind][label]
    # if sum(score)==0:
    #     cluterss=hdbscan.HDBSCAN(min_cluster_size=2,min_samples=1,metric='l2',prediction_data=True)
    #     cluterss.fit(fc3_vector_norm)
    #     cluterss_labels=cluterss.labels_
    #     soft_clusters = hdbscan.all_points_membership_vectors(cluterss)
    #     score=np.zeros(len(updates))
    #     for ind,label in enumerate(cluterss_labels):
    #         if label != -1:
    #             #score[label][ind]=soft_clusters[ind][label]
    #             score[ind]=soft_clusters[ind][label]
    return local_score


def ClusterGlobelVector(updates,net_glob):
    n=len(updates)
    k=28
    model_vector=[torch.nn.utils.parameters_to_vector(update.parameters()) \
        for update in [torch.nn.Sequential(*list(update.children())[:-1]) for update in updates]]
    model_vector_=torch.empty([n, len(model_vector[0])])
    for i in range(n): 
        model_vector_[i]=model_vector[i]
    model_vector_=model_vector_.detach().numpy()
    slect_vecter=[]
    for i in range(len(model_vector_)):
        slect_vecter.append(list(np.random.choice(model_vector_[i],int(0.1*len(model_vector_[i])))))
    #cdist=euclidean_distances(slect_vecter)
    slect_vecter=get_pca(slect_vecter)
    cluterss=hdbscan.HDBSCAN(min_cluster_size=80,min_samples=1,metric='euclidean',allow_single_cluster=True)
    cluterss.fit(slect_vecter)
    #soft_clusters = hdbscan.all_points_membership_vectors(cluterss)
    cluterss_labels=cluterss.labels_
    #clter=np.argmax(np.bincount(cluterss_labels))
    global_score=np.zeros(len(updates))
    for ind,label in enumerate(cluterss_labels):
        if label != -1:
            global_score[ind]=1      
    # dist, idxs = torch.topk(torch.tensor(cdist), k, largest=False)
    # dist = dist.sum(1)
    # idxs = dist.argsort()
    return global_score

#

##################################################################
def Repeated_Median_Shard(w, w_local):
    SHARD_SIZE = 100000
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y
    for k in w_med.keys():
        w_local[k] = w_med[k]
    return w_med, w_local


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output


class FLDefender:
    def __init__(self, num_peers):
        self.grad_history = None
        self.fgrad_history = None
        self.num_peers = num_peers
        self.score_history = np.zeros([self.num_peers], dtype=float)

    def score(self, global_model, local_models, peers_types, selected_peers, epoch, tau=1.5):
        global_model = list(global_model.parameters())
        last_g = global_model[-2].cpu().data.numpy()
        m = len(local_models)
        f_grads = [None for i in range(m)]
        for i in range(m):
            grad = (last_g - list(local_models[i].parameters())[-2].cpu().data.numpy())
            f_grads[i] = grad.reshape(-1)

        cs = smp.cosine_similarity(f_grads) - np.eye(m)
        cs = get_pca(cs)
        centroid = np.median(cs, axis=0)
        scores = smp.cosine_similarity([centroid], cs)[0]

        return scores

    def update_score_history(self, scores, selected_peers, epoch):
        logger.info('-> Update score history')
        self.score_history[selected_peers] += scores
        q1 = np.quantile(self.score_history, 0.25)
        trust = self.score_history - q1
        trust = trust / trust.max()
        trust[(trust < 0)] = 0
        return trust[selected_peers]


def get_pca(data):
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    return data


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers

    def score_gradients(self, global_model, local_models, selectec_peers):
        global_model = list(global_model.parameters())
        m = len(local_models)
        grads = [None for i in range(m)]
        for i in range(m):
            grads[i] = (global_model[-2].cpu().data.numpy() - list(local_models[i].parameters())[
                -2].cpu().data.numpy()).reshape(-1)

        grads = np.asarray(grads)
        if self.memory is None:
            self.memory = np.zeros((self.num_peers, len(grads[0])))

        self.memory[selectec_peers] += grads
        wv = foolsgold(self.memory)  # Use FG
        self.wv_history.append(wv)
        return wv[selectec_peers]


def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


def simple_median(w, w_local):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    for k in w_med.keys():
        w_local[k] = w_med[k]
    return w_med, w_local


def trimmed_mean(w, trim_ratio, w_local):
    if trim_ratio == 0:
        return average_weights(w, [1 for i in range(len(w))])

    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight

    return w_med
