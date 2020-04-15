import torch
import torch.nn as nn
import numpy as np
import argparse

from GAP import GAP, Generator, Discriminator
from utils import MyDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help="batch size during training", type=int, default=32)
args = parser.parse_args()

ngpu = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# TODO: import dataset here
input_data = None
label_data = None
dataset = MyDataset(input_data, label_data)
DataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
gap = GAP()
