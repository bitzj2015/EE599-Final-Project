import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init

class Gen_args(object):
    def __init__(self, 
                 vocab_size=3000, 
                 emb_dim=300, 
                 enc_hid_dim=64,
                 dec_hid_dim=64,
                 enc_dropout=0.5,
                 attn_dim=64,
                 dec_dropout=0.5):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_dropout = enc_dropout
        self.attn_dim = attn_dim
        self.dec_dropout = dec_dropout
        self.attn_in_dim = (self.enc_hid_dim * 2) + self.dec_hid_dim

class Generator(nn.Module):
    '''
    Generator
    '''
    def __init__(self, G_args, use_cuda=False):
        super(Generator, self).__init__()
        self.args = G_args
        self.use_cuda = use_cuda
        # Encoder
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.emb_dim)
        self.enc_rnn = nn.GRU(self.args.emb_dim, self.args.enc_hid_dim, bidirectional = True, batch_first=True)
        self.enc_fc = nn.Linear(self.args.enc_hid_dim * 2, self.args.dec_hid_dim)
        self.enc_dropout = nn.Dropout(self.args.enc_dropout)
        # Attention
        self.attn = nn.Linear(self.args.attn_in_dim, self.args.attn_dim)
        # Decoder
        self.dec_rnn = nn.GRU((self.args.enc_hid_dim * 2) + self.args.emb_dim, self.args.dec_hid_dim, batch_first=True)
        self.dec_out = nn.Linear(self.args.attn_in_dim + self.args.emb_dim, self.args.vocab_size)
        self.dec_dropout = nn.Dropout(self.args.dec_dropout)
        self.apply(weights_init)

    def encoder(self, input):
        # Encoder
        x = input[:,:,0]
        mask = input[:,:,1].float()
        emb = self.enc_dropout(self.embedding(x) * mask.unsqueeze(2))
        
        enc_out, enc_hid = self.enc_rnn(emb)
        enc_hid = torch.tanh(self.enc_fc(torch.cat((enc_hid[-2,:,:], enc_hid[-1,:,:]), dim = 1)))
        return enc_out, enc_hid
    
    def attention(self,
                  dec_hid,
                  enc_out):

        src_len = enc_out.shape[1]
        repeated_dec_hid = dec_hid.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((
            repeated_dec_hid,
            enc_out),
            dim = 2)))
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)

    def _weighted_enc_rep(self,
                          dec_hid,
                          enc_out):
        a = self.attention(dec_hid, enc_out)
        a = a.unsqueeze(1)
        # enc_out = enc_out.permute(1, 0, 2)
        weighted_enc_rep = torch.bmm(a, enc_out)
        # weighted_enc_rep = weighted_enc_rep.permute(1, 0, 2)
        return weighted_enc_rep

    def decoder(self,
                input,
                dec_hid,
                enc_out):
        input = input.unsqueeze(1)
        emb = self.dec_dropout(self.embedding(input))
        weighted_enc_rep = self._weighted_enc_rep(dec_hid,
                                                  enc_out)
        rnn_input = torch.cat((emb, weighted_enc_rep), dim = 2)
        dec_out, dec_hid = self.dec_rnn(rnn_input, dec_hid.unsqueeze(0))
        emb = emb.squeeze(1)
        dec_out = dec_out.squeeze(1)
        weighted_enc_rep = weighted_enc_rep.squeeze(1)
        dec_out = self.dec_out(torch.cat((dec_out,
                                     weighted_enc_rep,
                                     emb), dim = 1))
        return dec_out, dec_hid.squeeze(0)

    def forward(self,
                input):
        """
        Args:
            x: (batch_size, seq_len, 2), sequence of tokens generated by generator
        """
        batch_size = input.size(0)
        max_seq_len = input.size(1)
        outputs = torch.zeros(batch_size, max_seq_len, self.args.vocab_size)
        output = torch.zeros((batch_size)).long()
        if self.use_cuda:
            outputs = outputs.cuda()
            output = output.zeros()
        enc_out, hidden = self.encoder(input)
        outputs[:, 0] = torch.cat([torch.ones(batch_size,1), torch.zeros(batch_size, self.args.vocab_size - 1)], axis=1)
        # first input to the decoder is the <sos> token
        for t in range(1, max_seq_len):
            output, hidden = self.decoder(output, hidden, enc_out)
            outputs[:, t] = output
            top1 = output.max(1)[1]
            output = top1
        outputs = F.log_softmax(outputs.contiguous().view(-1, self.args.vocab_size), dim=1)
        return outputs

    def step(self, input, h, c):
        """
        Args:
            x: (batch_size,  1, 2), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        x = input[:, 0, 0]
        mask = input[:, 0, 1].float()
        emb = self.emb(x) * mask.unsqueeze(2)
        output, (h, c) = self.lstm(emb, (h, c))
        pred = F.softmax(self.fc(output.view(-1, self.args.hidden_dim)), dim=1)
        return pred, h, c

    def init_hidden(self, batch_size):
        h = torch.zeros((1, batch_size, self.args.hidden_dim))
        c = torch.zeros((1, batch_size, self.args.hidden_dim))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    def sample(self, batch_size, x_gen, target):
        flag = False # whether sample from zero
        if x_gen is None:
            flag = True
        if self.use_cuda:
            if not flag:
                x_gen = x_gen.cuda()
            target = target.cuda()
        if flag:
            output = self.forward(target)
            samples = torch.exp(output).multinomial(1).view(batch_size, -1)
        else:
            given_len = x_gen.size(1)
            output = self.forward(target[:, given_len:,: ])
            samples = [x_gen, torch.exp(output).multinomial(1).view(batch_size, -1)]
            samples = torch.cat(samples, dim=1)
        return samples, output

        