# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg), 
# FedAvg (--alg fedavg) and FedProx (--alg prox)

import copy
import itertools
import numpy as np
import pandas as pd
import torch
# import tqdm
from torch import nn
from loguru import logger
from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate
from models.test import test_img_local_all
# import heartrate; heartrate.trace(browser=True)
import time
from models.EdgeNode import EdgeNode
from utils.agg_methods import average_weights, average_weights_rep
from utils.defense_methods import trimmed_mean, Krum, simple_median, Repeated_Median_Shard, FLDefender, FoolsGold, Ours,ClusterLocalVector,ClusterGlobelVector,ClusterAVG

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        if args.iid==0:
            dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
            for idx in dict_users_train.keys():
                np.random.shuffle(dict_users_train[idx])
        else:
            dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    else:
        if 'femnist' in args.dataset:
            train_path = './leaf-master/data/' + args.dataset + '/data/mytrain'
            test_path = './leaf-master/data/' + args.dataset + '/data/mytest'
        else:
            train_path = './leaf-master/data/' + args.dataset + '/data/train'
            test_path = './leaf-master/data/' + args.dataset + '/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        print(lens)
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    print(args.alg)

    # build model
    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    if args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 3, 4]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 3,4]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
        else:
            w_glob_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1, 2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2, 3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 6, 7]]
        else:
            w_glob_keys = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox' or args.alg == 'maml':
        w_glob_keys = []
    if 'sent140' not in args.dataset:
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    # 将模型的部分权重(即rep)发送到全局模型（w_glob_keys）（femnist和mnist是全连接），net_glob为全部模型
    logger.info('模型总层数 {}'.format(total_num_layers))  # 模型总层数
    logger.info('模型rep层 {}'.format(w_glob_keys))  # 模型rep层
    logger.info('模型全部层 {}'.format(net_keys))  # 模型全部层
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()  # 计算总参数量
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()  # rep层参数量
        logger.info("模型本地总参数量 {}".format(num_param_local))
        percentage_param = 100 * float(num_param_glob) / num_param_local  # rep层参数量/总参数量
        logger.info('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    logger.info("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict  # 给每一个用户全部模型
    # 攻击者挑选
    n = max(int(args.num_users * args.attacker_rate), 1)
    num_users = dict()
    attacker = np.random.choice(range(args.num_users), n, replace=False)
    for i in range(args.num_users):
        num_users[i] = 'honest'
    if args.attack:
        for i in attacker:
            num_users[i] = 'attacker'

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    agg_times = []
    test_times = []
    local_weights = []
    local_models = []
    accs10 = 0
    accs10_glob = 0
    start = time.perf_counter()
    for iter in range(args.epochs + 1):
        local_weights = []
        local_models = []
        edge_weights = []
        edge_models = []
        if iter != args.epochs:
            logger.info("---------------------轮次{}开始------------------------".format(iter))
        else:
            logger.info("---------------------最后一轮开始-----------------------")
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 20)  # 客户端数量*选择百分比，最少是一个
        if iter == args.epochs:  # 最后一代
            m = args.num_users  # m为所有客户端

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 随机选择m个客户端
        w_keys_epoch = w_glob_keys  # 模型rep层
        times_in = []
        total_len = 0
        users_time = time.perf_counter()
        for ind, idx in enumerate(idxs_users):

            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]],
                                        idxs=dict_users_train, indd=indd)  # 最后一代进行头部微调，微调个数为m_ft
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]],
                                        idxs=dict_users_train, indd=indd)
            else:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft],
                                        peer_type=num_users[idx])
                else: 
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr],
                                        peer_type=num_users[idx])
            start_in = time.perf_counter()
            net_local = copy.deepcopy(net_glob)  # 复制全部网络模型，两个不同的对象
            w_local = net_local.state_dict()  # 全部参数
            if args.alg != 'fedavg' and args.alg != 'prox':
                for k in w_locals[idx].keys():  # 对于每个选中用户的模型
                    if k not in w_glob_keys:  # 选择头部微调模型
                        w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)  # 头部模型参数加载到全部模型中
            last = iter == args.epochs  # bools,1 or 0，是否为最后一轮训练
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx],
                                                  w_glob_keys=w_glob_keys, lr=args.lr, last=last)
            else:
                w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys,
                                                  lr=args.lr, last=last, net_glob=net_glob)
            loss_locals.append(copy.deepcopy(loss))

            w_local_dict = w_local.state_dict()
            # if len(w_glob) == 0:
            #     w_glob = copy.deepcopy(w_local_dict)
            #     for k, key in enumerate(net_glob.state_dict().keys()):
            #         w_glob[key] = w_glob[key] * lens[idx]
            #         w_locals[idx][key] = w_local_dict[key]
            # else:
            #     for k, key in enumerate(net_glob.state_dict().keys()):
            #         if key in w_glob_keys:
            #             w_glob[key] += w_local_dict[key] * lens[idx]
            #         else:
            #             w_glob[key] += w_local_dict[key] * lens[idx] ## lens/100个用户,100个1
            #         w_locals[idx][key] = w_local_dict[key]
            local_weights.append(w_local_dict)
            local_models.append(w_local)
            total_len += lens[idx]
            times_in.append(time.perf_counter() - start_in)
        logger.info("本地用户的训练平均时间 {}s".format((time.perf_counter() - users_time) / len(idxs_users)))
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        num_edge_nodes = 5
        clients_per_node = int(len(idxs_users) / num_edge_nodes)
        # 边缘节点,100*0.1 10个本地模型，可以分为五个边缘节点
        if args.edge == 1:
            edge_time_start = time.perf_counter()
            edge_nodes = EdgeNode(args, clients_per_node, num_edge_nodes, idxs_users)
            edge_nodes_model, edge_nodes_weight, edge_nodes_users = edge_nodes.edge_distribution(
                copy.deepcopy(local_models),
                copy.deepcopy(local_weights))
            updates_users = edge_nodes.edge_agg(edge_nodes_model, edge_nodes_weight, edge_nodes_users, w_glob_keys)
            logger.info("边缘节点模型过滤平均时间：{}ms".format(((time.perf_counter() - edge_time_start) * 1000) / 5))

        # 传输到服务器
        agg_time_start = time.perf_counter()
        #scores = np.zeros((num_edge_nodes,clients_per_node))
        scores = np.zeros(len(idxs_users))
        f = max(int(args.attacker_rate * len(local_weights)), 1)
        if args.rule == 'fedrep':
            w_glob, w_local_dict, w_locals = average_weights_rep(local_weights, [1 for i in range(len(local_weights))],
                                                                 net_glob, idxs_users, w_locals, w_glob_keys, w_glob)

        elif args.rule == 'fedavg':
            w_glob = average_weights(local_weights, [1 for i in range(len(local_weights))])
        elif args.rule == 'median': 
            w_glob, w_local_dict = simple_median(local_weights, w_local_dict)
        elif args.rule == 'rmedian':
            w_glob, w_local_dict = Repeated_Median_Shard(local_weights, w_local_dict)
        elif args.rule == 'fl_defender':
            scores = FLDefender(args.num_users).score(global_model=copy.deepcopy(net_glob),
                                                      local_models=copy.deepcopy(local_models),
                                                      peers_types=num_users[idx], selected_peers=idxs_users,
                                                      epoch=iter + 1, tau=(1.5 * iter / args.epochs))
            trust = FLDefender(args.num_users).update_score_history(scores, idxs_users, iter)
            w_glob = average_weights(local_weights, trust)
        elif args.rule == 'foolsgold':
            scores = FoolsGold(args.num_users).score_gradients(copy.deepcopy(net_glob),
                                                               copy.deepcopy(local_models),
                                                               idxs_users)
            w_glob = average_weights(local_weights, scores)
            #w_glob, w_local_dict, w_locals = average_weights_rep(local_weights, scores,
             #                                                    net_glob, idxs_users, w_locals, w_glob_keys, w_glob)
        elif args.rule == 'tmean':
            w_glob = trimmed_mean(local_weights, trim_ratio=args.attacker_rate, w_local=w_local_dict)
        elif args.rule == 'mkrum':
            goog_updates = Krum(local_models, f=f,net_glob=net_glob,multi=True)
            scores[goog_updates] = 1  # 20个参与训练的用户中选择
            #scores=ClusterAVG(local_models,net_glob,idxs_users)
            w_glob = average_weights(local_weights, scores)
            #w_glob, w_local_dict, w_locals = average_weights_rep(local_weights, scores,
               #                                                  net_glob, idxs_users, w_locals, w_glob_keys, w_glob)
            # w_glob = average_weights(local_weights,scores)
        elif args.rule == 'ours':
            # goog_updates = Krum(local_models, f=f,net_glob=net_glob,multi=True)
            # scores[goog_updates] = 1  # 20个参与训练的用户中选择
            scores=ClusterAVG(local_models,net_glob,idxs_users)
            # w_glob, w_local_dict = average_weights(local_weights, scores)
            w_glob, w_local_dict, w_locals = average_weights_rep(local_weights, scores,
                                                                 net_glob, idxs_users, w_locals, w_glob_keys, w_glob)
        agg_time = (time.perf_counter() - agg_time_start) * 1000
        logger.info("本轮次全局模型聚合时间：{}ms".format(agg_time))
        # get weighted average for global weights

        # for k in net_glob.state_dict().keys():
        #      w_glob[k] = torch.div(w_glob[k], total_len)

        # w_local_dict = net_glob.state_dict()
        # for k in w_glob.keys():
        #       w_local_dict[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)
        del local_weights
        del local_models
        agg_times.append(agg_time)
        if iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10:
            # logger.info("-----------开始测试-----------")
            test_time_start = time.perf_counter()
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                     w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
                                                     dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                     return_all=False)
            accs.append(acc_test)
            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs:
                logger.info('Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                logger.info('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))
            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10 += acc_test / 10

            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox':
                acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                         w_locals=None, indd=indd, dataset_train=dataset_train,
                                                         dict_users_train=dict_users_train, return_all=False)
                if iter != args.epochs:
                    logger.info(
                        'Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                            loss_avg, loss_test, acc_test))
                else:
                    logger.info(
                        'Final Round, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                            loss_avg, loss_test, acc_test))
            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10_glob += acc_test / 10
            test_time = time.perf_counter() - test_time_start
            logger.info("测试时间为：{}s".format(test_time))
            logger.info("---------------------结束----------------------------")
            test_times.append(test_time)
        if iter % args.save_every == args.save_every - 1:
            model_save_path = './save/accs_' + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) + '_iter' +'_'+args.rule+ str(iter) + '.pt'
            torch.save(net_glob.state_dict(), model_save_path)
            logger.info('Save Model: {}'.format(args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) + '_iter'  +'_'+args.rule+str(iter) + '.pt'))

    logger.info('Average accuracy final 10 rounds: {}'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        logger.info('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.perf_counter()
    logger.info("训练总时间: {}s".format(end - start))
    logger.info("用户训练总时间: {}s".format(times))
    logger.info("测试精度: {}".format(accs))
    base_dir_acc = './save/accs_' + args.alg + '_' + args.dataset + str(args.num_users) + '_' + str(
        args.shard_per_user)+'_'+args.rule + '.csv'
    user_save_path = base_dir_acc
    test_times = np.array(test_times)
    agg_times = np.array(agg_times)
    accs = np.array(accs)
    accs = pd.DataFrame({
        'accs': accs,
        'agg_times (ms)': agg_times,
        'test_times (s)': test_times
    })
    accs.to_csv(base_dir_acc, index=False)
