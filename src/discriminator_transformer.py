import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init, TransformerModel

class Dis_args(object):
    def __init__(self, 
                 vocab_size=3000, 
                 emb_dim=300, 
                 num_head=64,
                 hid_dim=64,
                 num_enc_l=0.5,
                 num_dec_l=64,
                 dropout=0.5,
                 out_dim=2):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.hid_dim = hid_dim
        self.num_enc_l = num_enc_l
        self.num_dec_l = num_dec_l
        self.dropout = dropout
        self.out_dim = out_dim

class Discriminator(nn.Module):
    '''
    Discriminator: a CNN text classifier
    '''
    def __init__(self, D_args, use_cuda=False):
        super(Discriminator, self).__init__()
        self.args = D_args
        self.use_cuda = use_cuda
        # Encoder
        self.transform = TransformerModel(self.args.vocab_size, 
                                          self.args.emb_dim,
                                          self.args.out_dim, 
                                          self.args.num_head, 
                                          self.args.hid_dim,
                                          self.args.num_enc_l,
                                          self.args.num_dec_l, 
                                          self.args.dropout)
        if self.use_cuda:
            self.transform = self.transform.cuda()
        self.apply(weights_init)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len, 2), sequence of tokens generated by generator
        """
        output, encoder = self.transform(input)
        pred = F.log_softmax(output, dim=2)
        seq_len = pred.size(1)
        out = pred.sum(1) / seq_len
        return out, pred, encoder
