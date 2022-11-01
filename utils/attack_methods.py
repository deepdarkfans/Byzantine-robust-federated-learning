import torch
from torch.utils import data
import numpy as np


class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class=None, target_class=None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class

    def __getitem__(self, index):
        x, y = self.dataset[index][0], self.dataset[index][1]
        if y == self.source_class:
            y = self.target_class
        return x, y

    def __len__(self):
        return len(self.dataset)


def label_filp(data, source_class, target_class):
    poisoned_data = PoisonedDataset(data, source_class, target_class)
    return poisoned_data


def gaussian_attack(update, malicious_behavior_rate=0, mean=0.0, std=200):
    flag = 0
    for key in update.keys():
        r = np.random.random()
        if r <= malicious_behavior_rate:
            # print('Gausiian noise attack launched by ', peer_pseudonym, ' targeting ', key, i+1)
            noise = torch.cuda.FloatTensor(update[key].shape).normal_(mean=mean, std=std)
            flag = 1
            update[key] = noise
    return update, flag
