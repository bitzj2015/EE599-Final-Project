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
                 hidden_dim=[3,4,5],nhead=6,nhid=2048,num_encode_layers=6,num_decode_layers=6,dropout=0.2):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.nhead=nhead
        self.nhid=nhid
        self.num_encode_layers=num_encode_layers
        self.num_decode_layers=num_decode_layers
        self.dropout=dropout

class Generator(nn.Module):
    '''
    Generator
    '''
    def __init__(self, G_args, use_cuda=False):
        super(Generator, self).__init__()
        self.args = G_args
        self.use_cuda = use_cuda
        # self.emb = nn.Embedding(self.args.vocab_size, self.args.emb_dim)
        #self.lstm = nn.LSTM(self.args.emb_dim, self.args.hidden_dim, batch_first=True)
        #self.fc = nn.Linear(self.args.hidden_dim, self.args.vocab_size)
        self.apply(weights_init)
        # self.transfm = nn.Transformer(d_model=self.args.emb_dim, self.args.hidden_dim, batch_first=True)
        self.transform = TransformerModel(self.args.vocab_size, self.args.emb_dim, self.args.nhead, self.args.nhid,self.args.num_encode_layers,self.args.num_decode_layers, self.args.dropout)
    
    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len, 2), sequence of tokens generated by generator
        """
        #mask = input[:,:,1].float()
        #emb = self.emb(x) * mask.unsqueeze(2)
        #h0, c0 = self.init_hidden(x.size(0))
        #output, (h, c) = self.lstm(emb, (h0, c0))
        #pred = F.softmax(self.fc(output.contiguous().view(-1, self.args.hidden_dim)), dim=1)

        x = input[:,:,0]
        output=self.transform(x)
        pred=F.log_softmax(output, dim=1)
        pred=pred.view(-1, self.args.vocab_size)

        return pred

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

    def sample(self, batch_size,target,x_gen=None):
        flag = False # whether sample from zero
        if x_gen is None:
            flag = True
        if self.use_cuda:
            if not flag:
                x_gen = x_gen.cuda()
            target = target.cuda()
        if flag:
            output = self.forward(target)
            samples = output.multinomial(1).view(batch_size, -1)
        else:
            given_len = x_gen.size(1)
            output = self.forward(target[:, given_len:,: ])
            samples = [x_gen, output.multinomial(1).view(batch_size, -1)]
        samples = torch.cat(samples, dim=1)
        return samples, output


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, num_encode_layers,num_decode_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoderLayer,TransformerDecoder
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encode_layers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        # decoder_layer = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        # self.decoder = TransformerDecoder(decoder_layer, num_decode_layers)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask


        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

        