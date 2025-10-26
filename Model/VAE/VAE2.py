import torch
import torch.nn as nn
import torch.optim as optim

# LSTM-based Variational Autoencoder for 1D Sequential Data

class LSTM_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        """
        input_dim  : Number of features per timestep
        hidden_dim : Hidden dimension of LSTM
        latent_dim : Latent space dimension
        num_layers : Number of stacked LSTM layers
        """
        super(LSTM_VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # ---------- Encoder ----------
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # ---------- Decoder ----------
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        returns: mu, logvar
        """
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]  # Take last hidden state
        mu = self.mu(h_last)
        logvar = self.logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        z = mu + eps * sigma
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        """
        z: [batch_size, latent_dim]
        seq_len: sequence length to reconstruct
        returns: reconstructed sequence
        """
        # Repeat latent vector for each timestep
        h_dec = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(h_dec)
        x_recon = self.output_layer(out)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, x.size(1))
        return x_recon, mu, logvar



# VAE Loss Function (Reconstruction + KL Divergence)

def vae_loss_function(x_recon, x, mu, logvar):
    recon_loss = nn.MSELoss()(x_recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss


# Training Function
def train_lstm_vae(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=10, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss_function(x_recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_recon, mu, logvar = model(x)
                loss, _, _ = vae_loss_function(x_recon, x, mu, logvar)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    print("Training complete.")

    return model
