import os
import random
import math
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Pool, Process, Queue
# from similarity import batch_similarity
import ray
ray.init()


class Rollout(object):
    '''
    Roll-out policy
    '''
    def __init__(self, model, update_rate, index_map):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.index_map = index_map

    def get_reward(self, x_gen, target, category, num, discriminator, adversary):
        '''
        Args:
            x_gen : (batch_size, seq_len) obfuscated data generated by generator, as input of discriminator
            target: (batch_size, seq_len, 2) original data, as conditional input for generating x_gen, for sampling
            num : roll-out number
            discriminator : discrimanator model
        '''
        batch_size = x_gen.size(0)
        seq_len = x_gen.size(1)

        @ray.remote
        def MonteCarlo(i, batch_size, model, x_gen, target, category, discriminator, adversary, index_map):
            print("[INFO] Process: {}".format(i))
            total_acc = 0.0
            sim_rewards = []
            dis_rewards = []
            adv_rewards = []
            for l in range(1, seq_len):
                data = x_gen[:, 0:l]
                samples, _ = model.sample(batch_size, data, target)
                samples_ = torch.stack([samples, target[:,:,1]], axis=2)
                dis_pred = discriminator(samples_).detach()
                dis_pred = torch.exp(dis_pred)[:,1]
                adv_pred = adversary(samples_).detach()
                adv_pred = torch.exp(torch.gather(adv_pred, 1, category.view(batch_size,1)).view(-1))
                adv_pred = 1 - adv_pred # batch_size
                sim_reward = torch.exp(1 - (seq_len - (samples == target[:,:,0]).sum(1).float())) * 1.0 / target[:,:,1].sum(1).float()
                sim_rewards.append(sim_reward)
                dis_rewards.append(dis_pred)
                adv_rewards.append(adv_pred)

            # for the last token
            samples_ = torch.stack([x_gen, target[:,:,1]], axis=2)
            dis_pred = discriminator(samples_).detach()
            dis_pred = torch.exp(dis_pred)[:,1]
            adv_pred = adversary(samples_).detach()
            _, pred_ = torch.max(adv_pred, axis=-1)
            total_acc += (pred_ == category).sum().item() / batch_size
            adv_pred = torch.exp(torch.gather(adv_pred, 1, category.view(batch_size,1)).view(-1))
            adv_pred = 1 - adv_pred
            sim_reward = torch.exp(1 - (seq_len - (samples == target[:,:,0]).sum(1).float()) * 1.0 / target[:,:,1].sum(1).float())
            sim_rewards.append(sim_reward)
            dis_rewards.append(dis_pred)
            adv_rewards.append(adv_pred)

            sim_rewards = torch.stack(sim_rewards, axis=1) # batch_size * seq_len
            dis_rewards = torch.stack(dis_rewards, axis=1) # batch_size * seq_len
            adv_rewards = torch.stack(adv_rewards, axis=1) # batch_size * seq_len
            return sim_rewards, dis_rewards, adv_rewards

        result = [MonteCarlo.remote(i, batch_size, self.own_model, \
            x_gen, target, category, discriminator, adversary, self.index_map) for i in range(num)]
        reward = ray.get(result)
        sim_avg_rewards = [item[0] for item in reward]
        dis_avg_rewards = [item[1] for item in reward]
        adv_avg_rewards = [item[2] for item in reward]
        return sum(sim_avg_rewards) / (1.0 * num), sum(dis_avg_rewards) / (1.0 * num), sum(adv_avg_rewards) / (1.0 * num)

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]