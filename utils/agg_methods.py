import copy
import torch


# Get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] * (1 / sum(marks))
    return w_avg


def average_weights_edge(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] * (1 / sum(marks))
    return w_avg


def average_weights_rep(local_weights, marks, net_glob, idxs_users, w_locals, w_glob_keys, w_glob):
    """
    Returns the average of the weights.
    """
    total_len = sum(marks)
    update=[]
    for ind, idx in enumerate(idxs_users):
        if len(w_glob) == 0:
            if marks[ind] != 0:
                w_glob = copy.deepcopy(local_weights[ind])
            # for ind, idx in enumerate(idxs_users):
                for k, key in enumerate(net_glob.state_dict().keys()):

                        w_glob[key] = w_glob[key] * marks[ind]

                        w_locals[idx][key] = local_weights[ind][key]
        else:
            for k, key in enumerate(net_glob.state_dict().keys()):
                if marks[ind] != 0:
                    if key in w_glob_keys:
                        w_glob[key] += local_weights[ind][key] * marks[ind]

                    else:
                        w_glob[key] += local_weights[ind][key] * marks[ind]

                    w_locals[idx][key] = local_weights[ind][key]
    for k in net_glob.state_dict().keys():
        w_glob[k] = torch.div(w_glob[k], total_len)
    w_local = net_glob.state_dict()
    for k in w_glob.keys():
        w_local[k] = w_glob[k]  # 这一句是关键
    return w_glob, w_local, w_locals
