import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init

class Dis_args(object):
    def __init__(self, 
                 vocab_size=3000, 
                 emb_dim=300, 
                 nhead=64,
                 nhid=64,
                 num_encode_layers=0.5,
                 num_decode_layers=64,
                 dropout=0.5):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.nhead = nhead
        self.nhid = nhid
        self.num_encode_layers = num_encode_layers
        self.num_decode_layers = num_decode_layers
        self.dropout = dropout

class Discriminator(nn.Module):
    '''
    Discriminator: a CNN text classifier
    '''
    def __init__(self, D_args):
        super(Discriminator, self).__init__()
        self.args = D_args
        self.emb = nn.Embedding(self.args.vocab_size, self.args.emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filter, (filter_size, self.args.emb_dim)) for (num_filter, filter_size) in zip(self.args.num_filters, self.args.filter_sizes)
        ])
        self.highway = nn.Linear(sum(self.args.num_filters), sum(self.args.num_filters))
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.fc = nn.Linear(sum(self.args.num_filters), self.args.num_classes)
        self.apply(weights_init)

    def forward(self, input):
        """
        Args:
            x: (batch_size * seq_len)
        """
        x = input[:,:,0]
        mask = input[:,:,1].float()
        emb = self.emb(x) * mask.unsqueeze(2)
        emb = emb.unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) *  F.relu(highway) + (1. - torch.sigmoid(highway)) * pred
        pred = F.log_softmax(self.fc(self.dropout(pred)), dim=1)
        return pred
