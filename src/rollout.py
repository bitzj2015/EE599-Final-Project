import os
import random
import math
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
        dis_rewards = []
        adv_rewards = []
        batch_size = x_gen.size(0)
        seq_len = x_gen.size(1)
        for i in tqdm(range(num)):
            # MC sampling times
            for l in range(1, seq_len):
                data = x_gen[:, 0:l]
                samples, _ = self.own_model.sample(batch_size, data, target)
                samples_ = torch.stack([samples, target[:,:,1]], axis=2)
                dis_pred = discriminator(samples_)
                dis_pred = dis_pred.cpu().data[:,1].numpy()
                adv_pred = adversary(samples_)
                adv_pred = np.exp(torch.gather(adv_pred, 1, category.view(batch_size,1)).view(-1).cpu().data.numpy())
                adv_pred = 1 - adv_pred # batch_size
                if i == 0:
                    dis_rewards.append(dis_pred)
                    adv_rewards.append(adv_pred)
                else:
                    dis_rewards[l-1] += dis_pred
                    adv_rewards[l-1] += adv_pred

            # for the last token
            samples_ = torch.stack([x_gen, target[:,:,1]], axis=2)
            dis_pred = discriminator(samples_)
            dis_pred = dis_pred.cpu().data[:, 1].numpy()
            adv_pred = adversary(samples_)
            adv_pred = np.exp(torch.gather(adv_pred, 1, category.view(batch_size,1)).view(-1).cpu().data.numpy())
            adv_pred = 1 - adv_pred
            if i == 0:
                dis_rewards.append(dis_pred)
                adv_rewards.append(adv_pred)
            else:
                dis_rewards[seq_len-1] += dis_pred
                adv_rewards[seq_len-1] += adv_pred
        dis_rewards = np.transpose(np.array(dis_rewards)) / (1.0 * num) # batch_size * seq_len
        adv_rewards = np.transpose(np.array(adv_rewards)) / (1.0 * num) # batch_size * seq_len
        return dis_rewards, adv_rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]