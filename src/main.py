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
from generator import Generator, Gen_args
from discriminator import Discriminator, Dis_args
from train import pretrain_gen, pretrain_adv
from data_loader import LoadData
from rollout import Rollout

# Set argument parser
parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', type=int, default=0)
parser.add_argument('--batch_size', help="batch size during training", type=int, default=64)
parser.add_argument('--phase', help="batch size during training", type=str, default="pretrain_G")
args = parser.parse_args()

# Set random seed
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# Basic Training Paramters
BATCH_SIZE = 64
VOCAB_SIZE = word_map["*"] + 1
USE_CUDA = args.cuda
PRE_EPOCH_NUM = 100
GEN_LR = 0.01
ADV_LR = 0.01
DIS_LR = 0.01
GEN_PATH = "../param/generator.pkl"
ADV_PATH = "../param/generator.pkl"
DIS_PATH = "../param/discriminator.pkl"

# Genrator Parameters
gen_args = Gen_args(vocab_size=VOCAB_SIZE, 
                    emb_dim=64, 
                    hidden_dim=64)

# Discriminator Parameters
dis_args = Dis_args(num_classes=2, 
                    vocab_size=VOCAB_SIZE, 
                    emb_dim=64, 
                    filter_sizes=[3, 4, 5], 
                    num_filters=[100, 100, 100], 
                    dropout=0.5)

# Adversarial Parameters
adv_args = Dis_args(num_classes=3, 
                    vocab_size=VOCAB_SIZE, 
                    emb_dim=64, 
                    filter_sizes=[3,4,5], 
                    num_filters=[100, 100, 100], 
                    dropout=0.5)

# Get training and testing dataloader
train_loader, test_loader, MAX_SEQ_LEN = LoadData(data_path="../data/dataset_batch.json", 
                                                  word2id_path="../data/word_map.json", 
                                                  train_split=0.8,
                                                  BATCH_SIZE=64)

# Define Networks
generator = Generator(gen_args, USE_CUDA)
discriminator = Discriminator(dis_args)
adversary = Discriminator(adv_args)
if USE_CUDA:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    adversary = adversary.cuda()



# Enter training phase
if args.phase == "pretrain_G":
    # Define optimizer and loss function for generator
    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters(), lr=GEN_LR)
    if USE_CUDA:
        gen_criterion = gen_criterion.cuda()
    # Pretrain generator using MLE
    pretrain_gen(generator, train_loader, test_loader, gen_criterion, gen_optimizer, GEN_PATH, USE_CUDA, PRE_EPOCH_NUM)

elif args.phase == "pretrain_A":
    # Define optimizer and loss function for adversarial
    adv_criterion = nn.NLLLoss(reduction='sum')
    adv_optimizer = optim.Adam(generator.parameters(), lr=ADV_LR)
    if USE_CUDA:
        adv_criterion = adv_criterion.cuda()
    # Pretrain adversary using CNN text classifier
    pretrain_adv(adversary, train_loader, test_loader, adv_criterion, adv_optimizer, ADV_PATH, USE_CUDA, PRE_EPOCH_NUM)
