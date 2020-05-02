from copy import deepcopy
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoderLayer,TransformerDecoder

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class MyDataset(Dataset):
    def __init__(self,
                 input_data,
                 label_data,
                 word_map,
                 max_len,
                 transform=None):
        self.input = input_data
        self.label = label_data
        self.word_map = word_map
        self.padding_value = word_map["*"]
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        data = self.input[idx]
        data = [self.word_map[token] for token in data]
        cur_len = len(data)
        data = np.array(data).astype('int')
        mask = data * 0 + 1
        data = np.pad(data, (0, self.max_len - cur_len), 'constant', \
                        constant_values=(0, self.padding_value)).reshape(-1)
        mask = np.pad(mask, (0, self.max_len - cur_len), 'constant', \
                        constant_values=(0, 0)).reshape(-1)
        data = np.stack([data,mask], axis=-1)
        user = self.label[idx]
        user = np.array(user).astype('int').reshape(-1)
        sample = {'x':data, 'u':user}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        data, user = sample['x'], sample['u']
        return {'x':torch.from_numpy(data), 'u':torch.from_numpy(user)}

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.BoolTensor)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss) / loss.size(0)
        return loss

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, num_words, emb_dim, out_dim, num_head, hid_dim, num_enc_l, num_dec_l, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        self.encoder_layers = TransformerEncoderLayer(emb_dim, num_head, hid_dim, dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_enc_l)
        self.encoder = nn.Embedding(num_words, emb_dim)
        self.emb_dim = emb_dim
        self.decoder = nn.Linear(emb_dim, out_dim)

        # decoder_layer = TransformerDecoderLayer(emb_dim, num_head, hid_dim, dropout)
        # self.decoder = TransformerDecoder(decoder_layer, num_dec_l)

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

    def forward(self, src, has_mask=True, USE_CUDA=False):

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask

        x = src[:,:,0]
        mask = src[:,:,1].float()
        src = self.encoder(x) * mask.unsqueeze(2)
        src = src * math.sqrt(self.emb_dim) 
        src = self.pos_encoder(src)
        if USE_CUDA:
            self.src_mask = self.src_mask.cuda()
        encoder = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(encoder)
        return output, encoder


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