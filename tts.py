import os
import librosa
import pandas as pd
import soundfile as sf
from fastdtw import fastdtw
import numpy as np
from filters import filter_text

# Пути к файлам
dataset_path = "data/prod/audio"
xlsx_path = "data/prod/Speeches.xlsx"

# Читаем данные из Excel
df = pd.read_excel(xlsx_path)

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

    for word in words:
        if word in text_to_wav:
            wav_file = text_to_wav[word]
            speech, sr, _ = audio_data[wav_file]
            speech_segments.append(speech)
        else:
            ref_wav_file = next(iter(audio_data))  # Первый доступный файл
            ref_audio, ref_sr, ref_mfcc = audio_data[ref_wav_file]

            text_mfcc = np.array(ref_mfcc)
            speech_segment = find_closest_match(text_mfcc, audio_data)
            speech_segments.append(speech_segment)

    full_speech = np.concatenate(speech_segments)
    sf.write(output_file, full_speech, 22050)

