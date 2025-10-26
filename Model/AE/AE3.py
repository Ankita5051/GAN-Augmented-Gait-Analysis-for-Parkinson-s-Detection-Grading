import torch
import torch.nn as nn

class LSTM_AE(nn.Module):
    def __init__(self, seq_len, latent_dim=32):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.hidden2latent = nn.Linear(in_features=64, out_features=latent_dim)
        self.latent2hidden = nn.Linear(in_features=latent_dim, out_features=64)
        self.decoder_lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.output = nn.Linear(64, 1)
        self.seq_len = seq_len

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        _, (h, _) = self.encoder_lstm(x)
        z = self.hidden2latent(h[-1])
        h_dec = self.latent2hidden(z).unsqueeze(0)
        dec_input = torch.zeros(x.size(0), self.seq_len, 1).to(x.device)
        out, _ = self.decoder_lstm(dec_input, (h_dec, torch.zeros_like(h_dec)))
        x_recon = self.output(out)
        return x_recon.squeeze(-1)
