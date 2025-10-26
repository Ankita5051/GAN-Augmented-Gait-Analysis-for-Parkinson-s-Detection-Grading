import torch
import os
import torch.nn as nn
import torch.optim as optim

noise_dim=int(os.environ['NOISE_DIM'])
lr_g=float(os.environ['LR_G'])
lr_d=float(os.environ['LR_D'])
class cGAN:
    def __init__(self, generator, discriminator, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    def train_step(self, real_data, labels):
        batch_size = real_data.size(0)

        # Labels for real and fake data
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # Train Discriminator

        self.optimizer_D.zero_grad()

        # Real data
        real_output = self.discriminator(real_data, labels)
        d_loss_real = self.criterion(real_output, real_labels)

        # Fake data
        noise = torch.randn(batch_size, noise_dim).to(self.device)
        fake_data = self.generator(noise, labels).detach()
        fake_output = self.discriminator(fake_data, labels)
        d_loss_fake = self.criterion(fake_output, fake_labels)

        # Total discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.optimizer_D.step()

        # Train Generator
        self.optimizer_G.zero_grad()

        noise = torch.randn(batch_size, noise_dim).to(self.device)
        fake_data = self.generator(noise, labels)
        fake_output = self.discriminator(fake_data, labels)

        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.optimizer_G.step()

        return d_loss.item(), g_loss.item()
