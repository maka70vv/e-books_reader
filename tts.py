import os

import librosa
import pandas as pd
import soundfile as sf
from fastdtw import fastdtw
import numpy as np

from filters import filter_text

# Пути к файлам
dataset_path = "data/test/normal_voices"
xlsx_path = "data/test/Speeches.xlsx"

# Читаем данные из Excel
df = pd.read_excel(xlsx_path)

# Загружаем аудиофайлы и их MFCC
audio_data = {}
for _, row in df.iterrows():
    wav_file = row["file"]
    text = row["text"]
    text = filter_text(text)
    audio_path = os.path.join(dataset_path, f"{wav_file}.wav")

    if os.path.exists(audio_path):
        y, sr = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        audio_data[text] = (y, sr, mfcc)


# Функция поиска ближайшего совпадения
def find_closest_match(text_mfcc, audio_data):
    best_match = None
    best_dist = float("inf")

    for text, (y, sr, mfcc) in audio_data.items():
        dist, _ = fastdtw(text_mfcc.T, mfcc.T)
        if dist < best_dist:
            best_dist = dist
            best_match = y

    return best_match


# Генерация речи
def generate_speech(text, audio_data, output_file):
    text = filter_text(text)
    # Разбиваем текст на слова
    words = text.split()

    # Массив для хранения аудио данных
    speech_segments = []

    for word in words:
        # Проверяем, есть ли слово в аудиоданных
        if word in audio_data:
            speech, sr, _ = audio_data[word]
            speech_segments.append(speech)
        else:
            # Если нет точного совпадения, ищем наиболее похожий фрагмент
            ref_text = list(audio_data.keys())[0]  # Выбираем любой реальный образец
            ref_audio, ref_sr, ref_mfcc = audio_data[ref_text]

            # Берем MFCC случайного реального аудиофайла
            text_mfcc = ref_mfcc
            speech_segment = find_closest_match(text_mfcc, audio_data)
            speech_segments.append(speech_segment)

    # Объединяем все фрагменты речи в один аудиофайл
    full_speech = np.concatenate(speech_segments)

    # Сохраняем результат
    sf.write(output_file, full_speech, 22050)
