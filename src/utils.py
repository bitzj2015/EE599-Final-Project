from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset

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