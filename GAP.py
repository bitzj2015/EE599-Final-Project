import torch
import torch.nn as nn
import numpy as np
from Model import TransformerModel

class Generator(nn.module):
    def __init__(self, name):
        super(Generator, self).__init__()
        self.name = name
        self.main = TransformerModel()
        self.fc = nn.Linear(512, 512)
    
    def forward(self, input):
        out = self.main(input)
        return self.fc(out)


class Discriminator(nn.module):
    def __init__(self, name):
        super(Discriminator, self).__init__()
        self.name = name
        self.main = TransformerModel()
        self.fc = nn.Linear(512, 512)
    
    def forward(self, input):
        out = self.main(input)
        return self.fc(out)

class GAP(object):
    def __init__(self, name, generator, discriminator, optimizer):
        self.name = name
        self.generator = generator
        self.discriminator = discriminator
        self.loss = None
        self.optimizer = optimizer

    def train(self, dataloader, max_epoch=10, load_model=None):
        if load_model != None:
            self.generator.load_state_dict(torch.load(load_model + "generator"))
            self.discriminator.load_state_dict(torch.load(load_model + "discriminator"))
        for ep in range(max_epoch):
            for i, batch in enumerate(dataloader):
                x = batch['x']

        