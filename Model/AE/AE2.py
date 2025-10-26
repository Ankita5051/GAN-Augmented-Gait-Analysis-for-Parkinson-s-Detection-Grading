import torch.nn as nn


class CNN1D_AE(nn.Module):
    def __init__(self,in_ch):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=in_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon.squeeze(1)
