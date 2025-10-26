import torch
import os
import torch.nn as nn
import torch.optim as optim

GP_WEIGHT=int(os.environ['GP_WEIGHT'])
NOISE_DIM=int(os.environ['NOISE_DIM'])
lr_g=int(os.environ['LR_G'])
lr_d=int(os.environ['LR_D'])
CRITIC_ITERATIONS = int(os.environ['CRITIC_ITERATIONS'])
# cWGAN
class cWGAN_GP:
    def __init__(self, generator, discriminator, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    def gradient_penalty(self, real_data, fake_data, labels):
        batch_size = real_data.size(0)

        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, 1).to(self.device)

        # Reshape real and fake data for broadcasting with alpha
        real_data = real_data.unsqueeze(-1)
        fake_data = fake_data.unsqueeze(-1)

        # Interpolated data
        interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

        # Discriminator output for interpolated data
        d_interpolated = self.discriminator(interpolated.squeeze(-1), labels) # Remove the added dimension for discriminator input

        # Gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def train_step(self, real_data, labels):
        batch_size = real_data.size(0)

        # Train Discriminator
        for _ in range(CRITIC_ITERATIONS):
            self.optimizer_D.zero_grad()

            # Real data
            real_output = self.discriminator(real_data, labels)
            d_loss_real = -torch.mean(real_output)

            # Fake data
            noise = torch.randn(batch_size, NOISE_DIM).to(self.device)
            fake_data = self.generator(noise, labels).detach()
            fake_output = self.discriminator(fake_data, labels)
            d_loss_fake = torch.mean(fake_output)

            # Gradient penalty
            gp = self.gradient_penalty(real_data, fake_data, labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake + GP_WEIGHT * gp
            d_loss.backward()
            self.optimizer_D.step()

        # Train Generator
        self.optimizer_G.zero_grad()

        noise = torch.randn(batch_size, NOISE_DIM).to(self.device)
        fake_data = self.generator(noise, labels)
        fake_output = self.discriminator(fake_data, labels)

        g_loss = -torch.mean(fake_output)
        g_loss.backward()
        self.optimizer_G.step()

        return d_loss.item(), g_loss.item()