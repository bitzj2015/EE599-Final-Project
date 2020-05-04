# Load basic module
import os
import json
import random
random.seed(0)
import math
from copy import deepcopy
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import GANLoss


# Load self-defined module
from generator import Generator, Gen_args
from discriminator import Discriminator, Dis_args
from train_seq import test_adv_epoch
from data_loader import LoadData

# Set random seed
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# Basic Training Paramters
BATCH_SIZE = 64
USE_CUDA = False
PRE_GEN_EPOCH_NUM = 20
PRE_ADV_EPOCH_NUM = 10
PRE_DIS_EPOCH_NUM = 2
GAP_EPOCH_NUM = 100
MC_NUM = 16
W = [0.02, 0.2, 0.8]
GEN_LR = 0.01
ADV_LR = 0.01
DIS_LR = 0.01
v = "W282"

PRE_GEN_PATH = "../param/pre_generator_v1.pkl"
PRE_ADV_PATH = "../param/pre_adversary_v1.pkl"
PRE_DIS_PATH = "../param/pre_discriminator_v1.pkl"

GEN_PATH = "../param/generator_v1-W282.pkl"
ADV_PATH = "../param/adversary_v1-W282.pkl"
DIS_PATH = "../param/discriminator_v1-W282.pkl"

# Get training and testing dataloader
train_loader, test_loader, \
    MAX_SEQ_LEN, VOCAB_SIZE, index_map = LoadData(data_path="../data/dataset_batch_v4.json", 
                                                  word2id_path="../data/word_map_v4.json", 
                                                  train_split=0.8,
                                                  BATCH_SIZE=64)


# Genrator Parameters
gen_args = Gen_args(vocab_size=VOCAB_SIZE, 
                    emb_dim=64, 
                    hidden_dim=64)

# Discriminator Parameters
dis_args = Dis_args(num_classes=2, 
                    vocab_size=VOCAB_SIZE, 
                    emb_dim=64, 
                    filter_sizes=[3, 4, 5], 
                    num_filters=[150, 150, 150], 
                    dropout=0.5)

# Adversarial Parameters
adv_args = Dis_args(num_classes=3, 
                    vocab_size=VOCAB_SIZE, 
                    emb_dim=64, 
                    filter_sizes=[3, 4, 5], 
                    num_filters=[150, 150, 150], 
                    # filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                    # num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                    dropout=0.5)


generator = Generator(gen_args, USE_CUDA)
discriminator = Discriminator(dis_args)
adversary = Discriminator(adv_args)

if USE_CUDA:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    adversary = adversary.cuda()

# Enter training phase
generator.load_state_dict(torch.load(GEN_PATH))
discriminator.load_state_dict(torch.load(DIS_PATH))
adversary.load_state_dict(torch.load(ADV_PATH))
gen_criterion = nn.NLLLoss(reduction='sum')

step = 0

gen_sim_loss = GANLoss()
gen_dis_loss = GANLoss()
gen_adv_loss = GANLoss()
W = torch.Tensor(W)
if USE_CUDA:
    gen_sim_loss = gen_sim_loss.cuda()
    gen_dis_loss = gen_dis_loss.cuda()
    gen_adv_loss = gen_adv_loss.cuda()
    W = W.cuda()

dis_reward_bias = 0
adv_reward_bias = 0
actual_adv_acc = 0
for batch in tqdm(test_loader): 
    step += 1  
    data, category = batch['x'], batch['u'].squeeze()
    if USE_CUDA:
        data = data.cuda()
        category = category.cuda()
    batch_size = data.size(0)
    target = data
    with torch.no_grad():
        samples, pred = generator.sample(batch_size, x_gen=None, target=target)
        samples_ = torch.stack([samples, target[:,:,1]], axis=2)
        dis_pred = discriminator(samples_).detach()
        dis_pred = torch.exp(dis_pred)[:,1]
        dis_reward_bias += dis_pred.data.cpu().numpy().sum() / batch_size
        adv_out = adversary(samples_).detach()
        _, adv_cat = torch.max(adv_out, dim=-1)
        adv_pred = torch.exp(torch.gather(adv_out, 1, category.view(batch_size,1)).view(-1))
        adv_pred = 1 - adv_pred    
        adv_reward_bias += adv_pred.data.cpu().numpy().sum() / batch_size
        actual_adv_acc += (adv_cat == category).sum().item() / batch_size

dis_reward_bias /= step
adv_reward_bias /= step
actual_adv_acc /= step

print("[INFO] (Dis_R, Adv_R): ({}, {}), adv_acc: {}".format(dis_reward_bias, adv_reward_bias, actual_adv_acc)) 

orig = data[:,:,0].detach().cpu().numpy()
obfu = samples.detach().cpu().numpy()
for i in range(batch_size):
    target_query = []
    pred_query = []
    for j in range(MAX_SEQ_LEN):
        target_query.append(index_map[orig[i][j]])
        pred_query.append(index_map[obfu[i][j]])
    print("[INFO] Target query: , ", target_query, "target category: ", category.data.cpu().numpy()[i])
    print("[INFO] Predicted query: ", pred_query, "pred category: ", adv_cat.data.cpu().numpy()[i])