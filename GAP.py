'''
reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Model import TransformerModel

class Adversarial(nn.module):
    def __init__(self, name):
        super(Adversarial, self).__init__()
        self.name = name
        self.main = TransformerModel()
        self.fc = nn.Linear(512, 512)
    
    def forward(self, input):
        out = self.main(input)
        return self.fc(out)

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
    def __init__(self, name, generator, discriminator, adversarial, lr=0.001, device="cpu"):
        self.name = name
        self.G = generator
        self.D = discriminator
        self.A = adversarial
        self.loss = None
        self.optimizerG = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerA = optim.Adam(self.A.parameters(), lr=lr, betas=(0.5, 0.999))
        self.device = device

    def train(self, dataloader, max_epoch=10, lr=0.001, load_model=None):
        # load model parameters
        if load_model != None:
            self.generator.load_state_dict(torch.load(load_model + "G.pkl"))
            self.discriminator.load_state_dict(torch.load(load_model + "D.pkl"))
        # change learning rate
        for param in self.optimizerD.param_groups:
            param['lr'] = lr
        for param in self.optimizerG.param_groups:
            param['lr'] = lr
        # real label = 1, fake label = 0
        G_losses = []
        D_losses = []
        A_losses = []
        real_label = 1
        fake_label = 0
        for ep in range(max_epoch):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(x)))
                ###########################
                ## Train with all-real batch
                self.D.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=device)
                # Forward pass real batch through D
                output = self.G(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.loss(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Generate fake batch with G
                data_ = self.G(data)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.D(data_.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.loss(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_x1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update A network: maximize loss(A(G(x)))
                # TODO: define the loss function of A(G(x))
                ###########################
                # Generate fake batch with G
                data_ = self.G(data)
                label.fill_(fake_label)
                # Classify all fake batch with A
                output = self.A(data_.detach()).view(-1)
                # Calculate A's loss on the all-fake batch
                # TODO
                errA = self.adversary_loss(output, data)
                # Calculate the gradients for this batch
                errA.backward()
                A_x = output.mean().item()
                # Update D
                self.optimizerA.step()

                ############################
                # (3) Update G network: maximize alpha * log(D(G(x))) + rho * utility_loss + (1-rho) * [-loss(A(G(x)))]
                ###########################
                self.G.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                output_D = self.D(data_).view(-1)
                errG_D = self.loss(output_D, label)
                # get adversarial loss
                output_A = self.A(data_.detach()).view(-1)
                errG_A = self.loss(output_A, label)
                # get utility loss
                errG_U = self.utility_loss(data, data_)
                # get errG
                errG = self.alpha * errG_D + self.rho * errG_U + (1 - self.rho) * (-errG_A)
                errG.backward()
                D_G_x2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_A: %.4f\tD(x): %.4f\tD(G(x)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(dataloader),
                            errD.item(), errG.item(), errA.item(), D_x, D_G_x1, D_G_x2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                A_losses.append(errA.item())
    
    def utility_loss(self, data, data_):
        # TODO: define utility loss here
        return True
    
    def adversary_loss(self, output, data):
        # TODO: define adversary loss here
        return True

        