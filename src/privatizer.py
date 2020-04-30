import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init

class Pri_args(object):
    def __init__(self, 
                 vocab_size=3000, 
                 emb_dim=300, 
                 enc_hid_dim=64,
                 dec_hid_dim=64,
                 enc_dropout=0.5):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_dropout = enc_dropout

class Privatizer(nn.Module):
    '''
    Privatizer
    '''
    def __init__(self, P_args, use_cuda=False):
        super(Privatizer, self).__init__()
        self.args = P_args
        self.use_cuda = use_cuda
        # Encoder
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.emb_dim)
        self.enc_rnn = nn.GRU(self.args.emb_dim, self.args.enc_hid_dim, bidirectional = True, batch_first=True)
        self.enc_fc = nn.Linear(self.args.enc_hid_dim * 2, self.args.dec_hid_dim)
        self.enc_dropout = nn.Dropout(self.args.enc_dropout)
        self.apply(weights_init)

    def forward(self, input):
        # Encoder
        x = input[:,:,0]
        mask = input[:,:,1].float()
        emb = self.enc_dropout(self.embedding(x) * mask.unsqueeze(2))
        
        enc_out, enc_hid = self.enc_rnn(emb)
        enc_hid = torch.tanh(self.enc_fc(torch.cat((enc_hid[-2,:,:], enc_hid[-1,:,:]), dim = 1)))
        return enc_out, enc_hid
