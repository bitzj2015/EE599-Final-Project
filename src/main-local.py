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
from discriminator import Discriminator, Dis_args
from privatizer import Privatizer, Pri_args
from train import pretrain_gen, train_adv, train_dis, train_pri
from data_loader import LoadData

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
USE_CUDA = args.cuda
PRE_GEN_EPOCH_NUM = 50
PRE_ADV_EPOCH_NUM = 2
PRE_DIS_EPOCH_NUM = 2
GAP_EPOCH_NUM = 30
MC_NUM = 16
GAP_W = [0.0, 0.5, 0.0]
GEN_LR = 0.003
ADV_LR = 0.003
DIS_LR = 0.003
PRI_LR = 0.003
PRE_GEN_PATH = "../param/pre_generator_v3.pkl"
PRE_ADV_PATH = "../param/pre_adversary.pkl"
PRE_DIS_PATH = "../param/pre_discriminator.pkl"
PRE_PRI_PATH = "../param/pre_privatizer.pkl"

GEN_PATH = "../param/generator_v4.pkl"
ADV_PATH = "../param/adversary_v4.pkl"
DIS_PATH = "../param/discriminator_v4.pkl"
PRI_PATH = "../param/privatizer_v4.pkl"

# Get training and testing dataloader
train_loader, test_loader, \
    MAX_SEQ_LEN, VOCAB_SIZE, index_map = LoadData(data_path="../data/dataset_batch_v3.json", 
                                                  word2id_path="../data/word_map_v3.json", 
                                                  train_split=0.8,
                                                  BATCH_SIZE=64)


# Genrator Parameters
# gen_args = Gen_args(vocab_size=VOCAB_SIZE, 
#                     emb_dim=64, 
#                     hidden_dim=64)
gen_args = Gen_args(vocab_size=VOCAB_SIZE, 
                    emb_dim=64,
                    enc_hid_dim=64,
                    dec_hid_dim=64,
                    enc_dropout=0.5,
                    attn_dim=8,
                    dec_dropout=0.5)

# Privatizer Parameters
pri_args = Pri_args(vocab_size=VOCAB_SIZE, 
                    emb_dim=64,
                    enc_hid_dim=64,
                    dec_hid_dim=64,
                    enc_dropout=0.5)

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

# Define Networks
generator = Generator(gen_args, USE_CUDA)
discriminator = Discriminator(dis_args)
adversary = Discriminator(adv_args)
privatizer = Privatizer(pri_args)

if USE_CUDA:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    adversary = adversary.cuda()
    privatizer = privatizer.cuda()

# Enter training phase
if args.phase == "pretrain_gen":
    generator.load_state_dict(torch.load(PRE_GEN_PATH))
    # Define optimizer and loss function for generator
    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters(), lr=GEN_LR)
    if USE_CUDA:
        gen_criterion = gen_criterion.cuda()
    # Pretrain generator using MLE
    pretrain_gen(generator=generator, 
                 train_loader=train_loader, 
                 test_loader=test_loader, 
                 gen_criterion=gen_criterion, 
                 gen_optimizer=gen_optimizer,
                 index_map=index_map, 
                 MAX_SEQ_LEN=MAX_SEQ_LEN,
                 GEN_PATH=PRE_GEN_PATH, 
                 USE_CUDA=USE_CUDA, 
                 EPOCH_NUM=PRE_GEN_EPOCH_NUM,
                 PLOT=True)

elif args.phase == "pretrain_adv":
    # Define optimizer and loss function for adversarial
    adv_criterion = nn.NLLLoss(reduction='sum')
    adv_optimizer = optim.Adam(adversary.parameters(), lr=ADV_LR)
    if USE_CUDA:
        adv_criterion = adv_criterion.cuda()
    # Pretrain adversary using CNN text classifier
    train_adv(adversary=adversary, 
              generator=None,
              privatizer=None,
              train_loader=train_loader, 
              test_loader=test_loader, 
              adv_criterion=adv_criterion,
              adv_optimizer=adv_optimizer, 
              ADV_PATH=PRE_ADV_PATH, 
              USE_CUDA=USE_CUDA, 
              EPOCH_NUM=PRE_ADV_EPOCH_NUM,
              PHASE="pretrain",
              PLOT=True)

elif args.phase == "pretrain_dis":
    # Define optimizer and loss function for discriminator
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=DIS_LR)
    if USE_CUDA:
        dis_criterion = dis_criterion.cuda()
    # Pretrain discriminator using CNN text classifier
    train_dis(discriminator=discriminator, 
              generator=generator,
              privatizer=None,
              train_loader=train_loader,
              test_loader=test_loader,
              dis_criterion=dis_criterion,
              dis_optimizer=dis_optimizer,
              DIS_PATH=PRE_DIS_PATH,
              USE_CUDA=USE_CUDA, 
              EPOCH_NUM=PRE_DIS_EPOCH_NUM,
              PHASE="pretrain", 
              PLOT=False)
# elif args.phase == "train_gap":
#     # Load pretrained parameters
#     try:
#         generator.load_state_dict(torch.load(PRE_GEN_PATH))
#         discriminator.load_state_dict(torch.load(PRE_DIS_PATH))
#         adversary.load_state_dict(torch.load(PRE_ADV_PATH))
#     except:
#         print("[Err] No pretrained model!")
#     # Define optimizer and loss function for discriminator
#     gen_criterion = nn.NLLLoss(reduction='sum')
#     gen_optimizer = optim.Adam(generator.parameters(), lr=GEN_LR)
#     dis_criterion = nn.NLLLoss(reduction='sum')
#     dis_optimizer = optim.Adam(discriminator.parameters(), lr=DIS_LR)
#     adv_criterion = nn.NLLLoss(reduction='sum')
#     adv_optimizer = optim.Adam(adversary.parameters(), lr=ADV_LR)
#     if USE_CUDA:
#         gen_criterion = gen_criterion.cuda()
#         dis_criterion = dis_criterion.cuda()
#         adv_criterion = adv_criterion.cuda()
#     # Pretrain discriminator using CNN text classifier
#     train_gap(model=[generator, discriminator, adversary],
#               criterion=[gen_criterion, dis_criterion, adv_criterion],
#               optimizer=[gen_optimizer, dis_optimizer, adv_optimizer],
#               train_loader=train_loader,
#               test_loader=test_loader,
#               index_map=index_map,
#               PATH=[GEN_PATH, DIS_PATH, ADV_PATH],
#               USE_CUDA=USE_CUDA,
#               EPOCH_NUM=GAP_EPOCH_NUM,
#               MC_NUM=MC_NUM,
#               W=GAP_W)
elif args.phase == "train_pri":
    # Load pretrained parameters
    try:
        privatizer.load_state_dict(torch.load(PRI_PATH))
        # generator.load_state_dict(torch.load(PRE_GEN_PATH))
        discriminator.load_state_dict(torch.load(DIS_PATH))
        adversary.load_state_dict(torch.load(ADV_PATH))
    except:
        print("[Err] No pretrained model!")
    # for name, p in generator.named_parameters():
    #     if "enc" in name or "emb" in name:
    #         p.requires_grad = False
    #     else:
    #         print(name, p.requires_grad)
    # Define optimizer and loss function for discriminator
    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters(), lr=GEN_LR)
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=DIS_LR)
    adv_criterion = nn.NLLLoss(reduction='sum')
    adv_optimizer = optim.Adam(adversary.parameters(), lr=ADV_LR)
    pri_criterion = nn.NLLLoss(reduction='sum')
    pri_optimizer = optim.Adam(privatizer.parameters(), lr=PRI_LR)
    if USE_CUDA:
        gen_criterion = gen_criterion.cuda()
        dis_criterion = dis_criterion.cuda()
        adv_criterion = adv_criterion.cuda()
        pri_criterion = pri_criterion.cuda()
    # Pretrain discriminator using CNN text classifier
    train_pri(model=[generator, discriminator, adversary, privatizer],
              criterion=[gen_criterion, dis_criterion, adv_criterion, pri_criterion],
              optimizer=[gen_optimizer, dis_optimizer, adv_optimizer, pri_optimizer],
              train_loader=train_loader,
              test_loader=test_loader,
              index_map=index_map,
              PATH=[GEN_PATH, DIS_PATH, ADV_PATH, PRI_PATH],
              USE_CUDA=USE_CUDA,
              EPOCH_NUM=GAP_EPOCH_NUM,
              MC_NUM=MC_NUM,
              W=GAP_W)
