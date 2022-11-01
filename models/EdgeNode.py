import copy
import numpy as np
from utils.agg_methods import average_weights_edge
from utils.defense_methods import Ours
import math

class EdgeNode:

    def __init__(self, args, clients_per_node, num_edge_nodes, idx_users):
        self.edge_weight = []
        self.edge_model = []
        self.edge_users = []
        self.scores = None
        self.edge_nodes_weight = dict()
        self.edge_nodes_model = dict()
        self.edge_nodes_users = dict()
        self.clients_per_node = clients_per_node
        self.num_edge_nodes = num_edge_nodes
        self.args = args
        self.updates=[]
        self.f = max(int(math.ceil( clients_per_node * self.args.attacker_rate)),1)
        self.idx_users = idx_users
        for i in range(num_edge_nodes):
            self.edge_nodes_weight["node{}".format(i + 1)] = None
            self.edge_nodes_model["node{}".format(i + 1)] = None
            self.edge_nodes_users["node{}".format(i + 1)] = None

    def edge_distribution(self, local_models, local_weights):
        for i in range(0, len(local_models), int(self.clients_per_node)):
            self.edge_model.append(local_models[i:i + int(self.clients_per_node)])
            self.edge_weight.append(local_weights[i:i + int(self.clients_per_node)])
            self.edge_users.append(self.idx_users[i:i + int(self.clients_per_node)])
        for k in range(len(self.edge_model)):
            self.edge_nodes_model["node{}".format(k + 1)] = self.edge_model[k]
            self.edge_nodes_weight["node{}".format(k + 1)] = self.edge_weight[k]
            self.edge_nodes_users["node{}".format(k + 1)] = self.edge_users[k]
        return self.edge_nodes_model, self.edge_nodes_weight, self.edge_nodes_users

    def edge_agg(self, edge_nodes_model, edge_nodes_weight, edge_nodes_users,w_glob_keys):
        # self.scores = np.zeros(len(local_weights))
        # self.f = int(self.args.attacker_rate * len(local_weights))
        edge_agg_model = []
        edge_agg_weight = []
        for key in edge_nodes_model.keys():
            # edge_agg_weight.append(
            #     average_weights_edge(edge_nodes_weight[key], [1 for i in range(len(edge_nodes_weight[key]))]))
           self.updates.append(Ours(edge_nodes_model[key], f=self.f, w_glob_keys=w_glob_keys,multi=True))
        # for i in range(len(edge_agg_weight)):
        #     edge_agg_model.append(copy.deepcopy(edge_nodes_model))
        #     edge_agg_model[i].load_state_dict(edge_agg_weight[i])

        return self.updates
