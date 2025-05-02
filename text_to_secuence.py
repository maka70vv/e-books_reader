import librosa
import numpy as np
import torch
from filters import filter_text

SAMPLE_RATE = 22050
N_MFCC = 40

def text_to_sequence(text, input_dim=N_MFCC):
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)

    filtered_text = filter_text(text)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=input_dim)

    mfcc_tensor = torch.tensor(mfcc.T, dtype=torch.float32)
    return mfcc_tensor
