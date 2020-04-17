import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init

class D_args(object):
    def __init__(self, 
                 num_classes=2, 
                 vocab_size=3000, 
                 emb_dim=300, 
                 filter_sizes=[3,4,5], 
                 num_filters=[64,64,64], 
                 dropout=0.2):
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.filter_sizes = filter_sizes 
        self.num_filters = num_filters 
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
        self.softmax = nn.LogSoftmax()
        self.apply(weights_init)

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.softmax(self.fc(self.dropout(pred)))
        return pred
