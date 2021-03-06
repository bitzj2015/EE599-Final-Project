import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init

class Dis_args(object):
    def __init__(self, 
                 vocab_size=3000, 
                 emb_dim=300, 
                 enc_hid_dim=64,
                 dec_hid_dim=64,
                 enc_dropout=0.5,
                 attn_dim=64,
                 dec_dropout=0.5,
                 out_dim=2):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_dropout = enc_dropout
        self.attn_dim = attn_dim
        self.dec_dropout = dec_dropout
        self.attn_in_dim = (self.enc_hid_dim * 2) + self.dec_hid_dim
        self.out_dim = out_dim

class Discriminator(nn.Module):
    '''
    Discriminator
    '''
    def __init__(self, D_args, use_cuda=False):
        super(Discriminator, self).__init__()
        self.args = D_args
        self.use_cuda = use_cuda
        self.emb = nn.Embedding(self.args.vocab_size, self.args.emb_dim)
        self.lstm = nn.LSTM(self.args.emb_dim, self.args.enc_hid_dim, batch_first=True)
        
        self.fc = nn.Linear(self.args.enc_hid_dim, self.args.out_dim)
        self.apply(weights_init)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len, 2), sequence of tokens generated by generator
        """
        x = input[:,:,0]
        mask = input[:,:,1].float()
        emb = self.emb(x) * mask.unsqueeze(2)
        h0, c0 = self.init_hidden(x.size(0))
        self.lstm.flatten_parameters()
        output, (h, c) = self.lstm(emb, (h0, c0))
        pred = F.log_softmax(self.fc(output), dim=2) * mask.unsqueeze(2)
        return pred, pred.sum(1) / mask.unsqueeze(2).sum(1)

    def init_hidden(self, batch_size):
        h = torch.zeros((1, batch_size, self.args.enc_hid_dim))
        c = torch.zeros((1, batch_size, self.args.enc_hid_dim))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c