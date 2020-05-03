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
from generator_seq import Generator, Gen_args
from discriminator_seq import Discriminator, Dis_args
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
W = [0.04, 0.5, 0.5]
GEN_LR = 0.01
ADV_LR = 0.01
DIS_LR = 0.01
v = "test"

PRE_GEN_PATH = "../param/pre_generator_v2.pkl"
PRE_ADV_PATH = "../param/pre_adversary_v2.pkl"
PRE_DIS_PATH = "../param/pre_discriminator_v2.pkl"

GEN_PATH = "../param/generator_v2" + v + ".pkl"
ADV_PATH = "../param/adversary_v2" + v + ".pkl"
DIS_PATH = "../param/discriminator_v2" + v + ".pkl"

# Get training and testing dataloader
train_loader, test_loader, \
    MAX_SEQ_LEN, VOCAB_SIZE, index_map = LoadData(data_path="../data/dataset_batch_v3_.json", 
                                                  word2id_path="../data/word_map_v3_.json", 
                                                  train_split=0.8,
                                                  BATCH_SIZE=64)


# Genrator Parameters
gen_args = Gen_args(vocab_size=VOCAB_SIZE, 
                    emb_dim=64,
                    enc_hid_dim=64,
                    dec_hid_dim=64,
                    enc_dropout=0.5,
                    attn_dim=8,
                    dec_dropout=0.5)

# Discriminator Parameters
dis_args = Dis_args(vocab_size=VOCAB_SIZE, 
                    emb_dim=64,
                    enc_hid_dim=64,
                    dec_hid_dim=64,
                    enc_dropout=0.5,
                    attn_dim=8,
                    dec_dropout=0.5,
                    out_dim=2)

# Adversarial Parameters
adv_args = Dis_args(vocab_size=VOCAB_SIZE, 
                    emb_dim=64,
                    enc_hid_dim=64,
                    dec_hid_dim=64,
                    enc_dropout=0.5,
                    attn_dim=8,
                    dec_dropout=0.5,
                    out_dim=3)

generator = Generator(gen_args, USE_CUDA)
discriminator = Discriminator(dis_args, USE_CUDA)
adversary = Discriminator(adv_args, USE_CUDA)

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
total_gen_loss = 0
total_gen_mle_loss = 0
total_gen_dis_loss = 0
total_gen_adv_loss = 0
total_dis_acc = 0
total_adv_acc = 0
actual_adv_acc = 0

gen_sim_loss = GANLoss()
gen_dis_loss = GANLoss()
gen_adv_loss = GANLoss()
W = torch.Tensor(W)
if USE_CUDA:
    gen_sim_loss = gen_sim_loss.cuda()
    gen_dis_loss = gen_dis_loss.cuda()
    gen_adv_loss = gen_adv_loss.cuda()
    W = W.cuda()

for batch in tqdm(test_loader): 
    step += 1
    data, category = batch['x'], batch['u'].squeeze()
    batch_size = data.size(0)
    seq_len = data.size(1)
    dis_pred_label = torch.ones((batch_size)).long()
    if USE_CUDA:
        data, category, dis_pred_label = \
        data.cuda(), category.cuda(), dis_pred_label.cuda()

    with torch.no_grad():
        output = generator.forward(input=data)
        _, pred_ = torch.max(output, axis=-1)
        samples = pred_.view(batch_size, -1)
        samples_ = torch.stack([samples, data[:,:,1]], axis=2)
        dis_pred, dis_out = discriminator(samples_)
        adv_pred, adv_out = adversary(samples_)
        _, adv_cat = torch.max(adv_out, dim=-1)
        dis_acc = torch.exp(dis_out)[:,1].sum() / batch_size
        adv_acc = 1 - torch.exp(torch.gather(adv_out, 1, category.view(batch_size,1)).view(-1)).sum() / batch_size
        dis_r = torch.exp(dis_pred)[:,:,1]
        category_ = category.view(batch_size,1).unsqueeze(1).repeat(1, seq_len, 1)
        adv_r = 1 - torch.exp(torch.gather(adv_pred, 2, category_))
        
        dis_loss = gen_dis_loss(output, samples.contiguous().view(-1), dis_r.contiguous().view(-1))
        adv_loss = gen_adv_loss(output, samples.contiguous().view(-1), adv_r.contiguous().view(-1))
        mle_loss = gen_criterion(output, data[:, :, 0].contiguous().view(-1)) / (data.size(0) * data.size(1))
        gen_loss = W[0] * mle_loss + W[1] * dis_loss + W[2] * adv_loss

    total_gen_loss += gen_loss.item()
    total_gen_mle_loss += mle_loss.item()
    total_gen_dis_loss += dis_loss.item()
    total_gen_adv_loss += adv_loss.item()
    total_dis_acc += dis_acc.data.cpu().numpy()
    total_adv_acc += adv_acc.data.cpu().numpy()
    actual_adv_acc += (adv_cat == category).sum().item() / batch_size



print("[INFO] Loss: {}, mle_loss: {}, dis_loss: {}, adv_loss: {}, \
    dis_R: {}, adv_R: {}, adv_acc: {}".\
        format(total_gen_loss/step, total_gen_mle_loss/step, total_gen_dis_loss/step, \
            total_gen_adv_loss/step, total_dis_acc/step, total_adv_acc/step, actual_adv_acc/step))

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