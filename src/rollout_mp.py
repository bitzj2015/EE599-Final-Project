import os
import random
import math
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import shared_memory

class Rollout(object):
    '''
    Roll-out policy
    '''
    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

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
        total_acc = 0.0
        a = np.array([1, 1, 2, 3, 5, 8]) 
        for i in tqdm(range(num)):
        def MonteCarlo(model, x_gen, target, category, num, discriminator, adversary, dis_avg_rewards, adv_avg_rewards):
            dis_rewards = []
            adv_rewards = []
            for l in range(1, seq_len):
                data = x_gen[:, 0:l]
                samples, _ = self.own_model.sample(batch_size, data, target)
                samples_ = torch.stack([samples, target[:,:,1]], axis=2)
                dis_pred = discriminator(samples_).detach()
                dis_pred = torch.exp(dis_pred)[:,1]
                adv_pred = adversary(samples_).detach()
                adv_pred = torch.exp(torch.gather(adv_pred, 1, category.view(batch_size,1)).view(-1))
                adv_pred = 1 - adv_pred # batch_size
                if i == 0:
                    dis_rewards.append(dis_pred)
                    adv_rewards.append(adv_pred)
                else:
                    dis_rewards[l-1] += dis_pred
                    adv_rewards[l-1] += adv_pred

            # for the last token
            samples_ = torch.stack([x_gen, target[:,:,1]], axis=2)
            dis_pred = discriminator(samples_).detach()
            dis_pred = torch.exp(dis_pred)[:,1]
            adv_pred = adversary(samples_).detach()
            _, pred_ = torch.max(adv_pred, axis=-1)
            total_acc += (pred_ == category).sum().item() / batch_size
            # print("check1:", torch.exp(adv_pred)[0:10])
            adv_pred = torch.exp(torch.gather(adv_pred, 1, category.view(batch_size,1)).view(-1))
            # print("check2:", category.view(batch_size,1)[0:10])
            adv_pred = 1 - adv_pred
            if i == 0:
                dis_rewards.append(dis_pred)
                adv_rewards.append(adv_pred)
            else:
                dis_rewards[seq_len-1] += dis_pred
                adv_rewards[seq_len-1] += adv_pred
            dis_rewards = torch.stack(dis_rewards, axis=1) # batch_size * seq_len
            adv_rewards = torch.stack(adv_rewards, axis=1) # batch_size * seq_len
            dis_avg_rewards += dis_rewards
            adv_avg_rewards += adv_rewards
        # print("Total acc:",total_acc / num)
        return dis_avg_rewards / (1.0 * num), adv_avg_rewards / (1.0 * num)

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]