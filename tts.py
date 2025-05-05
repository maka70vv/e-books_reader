import os
import librosa
import pandas as pd
import soundfile as sf
from fastdtw import fastdtw
import numpy as np
import torch
from filters import filter_text
from model import AcousticModel
from text_to_secuence import text_to_sequence
from vocoder import Vocoder

# Пути к файлам
dataset_path = "data/prod/audio"
xlsx_path = "data/prod/Speeches.xlsx"

# Читаем данные из Excel
df = pd.read_excel(xlsx_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

acoustic_model = AcousticModel().to(device)
vocoder = Vocoder().to(device)

# Загружаем аудиофайлы и их MFCC
audio_data = {}
text_to_wav = {}  # Новый словарь для связи текста и файлов

for _, row in df.iterrows():
    wav_file = row["file"]
    text = row["text"]
    text = filter_text(text)
    audio_path = os.path.join(dataset_path, f"{wav_file}.wav")

    if os.path.exists(audio_path):
        y, sr = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        audio_data[wav_file] = (y, sr, mfcc)
        text_to_wav[text] = wav_file  # Сохраняем связь текст → файл


# Функция поиска ближайшего совпадения
def find_closest_match(text_mfcc, audio_data):
    best_match = None
    best_dist = float("inf")

    for wav_file, (y, sr, mfcc) in audio_data.items():
        dist, _ = fastdtw(text_mfcc.T, mfcc.T)
        if dist < best_dist:
            best_dist = dist
            best_match = y

    return best_match


# Генерация речи
def generate_speech(text, audio_data, output_file):
    text = filter_text(text)
    words = text.split()
    speech_segments = []

    acoustic_model.eval()
    vocoder.eval()

    for word in words:
        if word in text_to_wav:
            wav_file = text_to_wav[word]
            speech, sr, _ = audio_data[wav_file]
            speech_segments.append(speech)
        else:
            seq_tensor = text_to_sequence(word).unsqueeze(0).to(device)

            with torch.no_grad():
                predicted_mfcc = acoustic_model(seq_tensor)
                predicted_mfcc = predicted_mfcc.permute(0, 2, 1)

                predicted_audio = vocoder(predicted_mfcc)
                audio = predicted_audio.squeeze().cpu().numpy()
                audio = audio / np.max(np.abs(audio) + 1e-6)

                speech_segments.append(audio)

    full_speech = np.concatenate(speech_segments)
    sf.write(output_file, full_speech, 22050)

