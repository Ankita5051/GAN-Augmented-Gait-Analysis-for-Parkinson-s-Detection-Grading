import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE1D(nn.Module):
    def __init__(self, in_channels=1, seq_len=3000, latent_dim=32):
        super(VAE1D, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=2, padding=2),  # (B,16,L/2)
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),           # (B,32,L/4)
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),           # (B,64,L/8)
            nn.ReLU(),
        )

        # compute flattened feature size after conv layers dynamically
        test_input = torch.zeros(1, in_channels, seq_len)
        with torch.no_grad():
            conv_out = self.encoder(test_input)
        self.flatten_dim = conv_out.view(1, -1).size(1)

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # ---------- Decoder ----------
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=in_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        x = x.permute(0, 2, 1)  # (B, in_channels, seq_len)
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), 64, -1)  # reshape to (B,64,L/8)
        x_recon = self.decoder(h)
        return x_recon.permute(0, 2, 1)  # back to (B, seq_len, 1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar



def vae_loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL Divergence: encourages the latent space to be normal
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss
