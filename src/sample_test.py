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

# Load self-defined module
from generator_seq import Generator, Gen_args
from discriminator_seq import Discriminator, Dis_args
from train_seq import pretrain_gen, train_adv, train_dis, train_pri, train_gap
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
GAP_W = [0.01, 0.2, 0.8]
GEN_LR = 0.01
ADV_LR = 0.01
DIS_LR = 0.01
v = "12"

PRE_GEN_PATH = "../param/pre_generator_v2.pkl"
PRE_ADV_PATH = "../param/pre_adversary_v2.pkl"
PRE_DIS_PATH = "../param/pre_discriminator_v2.pkl"

GEN_PATH = "../param/generator_v2" + v + ".pkl"
ADV_PATH = "../param/adversary_v2" + v + ".pkl"
DIS_PATH = "../param/discriminator_v2" + v + ".pkl"

# Get training and testing dataloader
train_loader, test_loader, \
    MAX_SEQ_LEN, VOCAB_SIZE, index_map = LoadData(data_path="../data/dataset_batch_v3.json", 
                                                  word2id_path="../data/word_map_v3.json", 
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
gen_criterion = nn.NLLLoss(reduction='sum')

model = generator
criterion = gen_criterion
total_loss = 0.
total_words = 0.
for batch in tqdm(test_loader):
    data = batch["x"]
    target = batch["x"][:,:,0]
    if USE_CUDA:
        data, target = data.cuda(), target.cuda()
    target = target.contiguous().view(-1)
    with torch.no_grad():
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
target_ = target.detach().cpu().numpy()
_, pred_ = torch.max(pred, axis=-1)
pred_ = pred_.cpu().numpy()
target_query = []
pred_query = []
for i in range(MAX_SEQ_LEN * 10):
    target_query.append(index_map[target_[i]])
    pred_query.append(index_map[pred_[i]])
print("[INFO] Target query: ", target_query)
print("[INFO] Predicted query: ", pred_query)