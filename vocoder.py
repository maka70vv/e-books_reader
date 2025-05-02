import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model


class Vocoder(nn.Module):
    def __init__(self, input_dim=model.N_MFCC, output_dim=1):
        super(Vocoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Выход от -1 до 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
