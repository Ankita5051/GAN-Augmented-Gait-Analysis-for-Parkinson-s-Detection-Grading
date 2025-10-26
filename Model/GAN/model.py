import torch
import os
from dotenv import load_dotenv
import torch.nn as nn
load_dotenv()

seq_len=int(os.environ['WINDOW_SIZE'])
in_channel=int(os.environ['NUM_VARIABLES'])
num_classes=int(os.environ['NUM_CLASSES'])
noise_dim=int(os.environ['NOISE_DIM'])

# Generator Model
class Generator(nn.Module):
    def __init__(self, noise_dim=noise_dim, num_classes=num_classes, seq_len=seq_len):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.seq_len = seq_len

        # Initial size after first dense layer
        self.init_size = 26

        # Dense layer to expand noise + condition
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + num_classes, in_channel * self.init_size),
            nn.ReLU(True)
        )

        # Convolutional layers with upsampling
        self.conv_blocks = nn.Sequential(
            # Block 1: 26 -> 26
            nn.Conv1d(in_channel, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 26 -> 52

            # Block 2: 52 -> 52
            nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 52 -> 104

            # Block 3: 104 -> 104
            nn.Conv1d(512, 256, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            # Block 4: 104 -> 104
            nn.Conv1d(256, 128, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            # Block 5: 104 -> 104
            nn.Conv1d(128, 64, kernel_size=32, stride=1, padding=16),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 104 -> 208

            # Final layer: 208 -> 204
            nn.Conv1d(64, in_channel, kernel_size=5, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # One-hot encode labels
        batch_size = noise.size(0)
        labels_onehot = torch.zeros(batch_size, self.num_classes).to(noise.device)
        labels_onehot.scatter_(1, labels.long().view(-1, 1), 1)

        # Concatenate noise and labels
        gen_input = torch.cat([noise, labels_onehot], dim=1)

        # Dense layer
        x = self.fc(gen_input)
        x = x.view(batch_size, in_channel, self.init_size)

        # Convolutional blocks
        x = self.conv_blocks(x)

        # Interpolate to exact sequence length
        if x.size(2) != self.seq_len:
            x = nn.functional.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)

        # Transpose to (batch, seq_len, channels)
        x = x.transpose(1, 2)

        return x


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, num_classes=num_classes, seq_len=seq_len):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len

        # Convolutional feature extractor
        self.conv_blocks = nn.Sequential(
            # Input: (batch, 1, 3000)
            nn.Conv1d(in_channel, 96, kernel_size=48, stride=12),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(96, 256, kernel_size=8, stride=1, padding=4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, 512, kernel_size=6, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(512, 512, kernel_size=4, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channel, seq_len)
            dummy_output = self.conv_blocks(dummy_input)
            self.flat_size = dummy_output.view(1, -1).size(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(32, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(8, 1)
        )

    def forward(self, x, labels):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.size(2) == in_channel:
            x = x.transpose(1, 2)

        # Extract features
        features = self.conv_blocks(x)
        features = features.view(features.size(0), -1)

        # One-hot encode labels
        batch_size = labels.size(0)
        labels_onehot = torch.zeros(batch_size, self.num_classes).to(labels.device)
        labels_onehot.scatter_(1, labels.long().view(-1, 1), 1)

        # Concatenate features and labels
        combined = torch.cat([features, labels_onehot], dim=1)

        # Final classification
        output = self.fc(combined)

        return output
