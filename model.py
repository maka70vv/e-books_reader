import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence  # Для динамического паддинга

import tts
from filters import filter_text

# Гиперпараметры
SAMPLE_RATE = 22050
N_MFCC = 40
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

# Чтение данных из Excel
def load_transcriptions(excel_path):
    df = pd.read_excel(excel_path)
    transcriptions = {}

    for _, row in df.iterrows():
        wav_file = row["file"]  # Имя файла
        text = row["text"]  # Текстовая расшифровка
        text = filter_text(text)  # Фильтрация текста
        audio_path = os.path.join(tts.dataset_path, f"{wav_file}.wav")  # Путь к аудиофайлу

        if os.path.exists(audio_path):
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            transcriptions[wav_file] = (text, mfcc)

    return transcriptions


# Подготовка данных
class TTSDataset(Dataset):
    def __init__(self, dataset_path, transcriptions):
        self.dataset_path = dataset_path
        self.transcriptions = transcriptions
        self.audio_files = list(transcriptions.keys())

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_name = self.audio_files[idx]
        text, _ = self.transcriptions[file_name]
        audio_path = os.path.join(self.dataset_path, f"{file_name}.wav")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        mfcc = torch.tensor(mfcc.T, dtype=torch.float32)  # Транспонируем MFCC

        return mfcc, y  # Возвращаем MFCC и оригинальный аудиосигнал


# Функция для динамического паддинга внутри DataLoader
def collate_fn(batch):
    mfccs, y = zip(*batch)  # Разбираем батч
    mfccs = pad_sequence(mfccs, batch_first=True, padding_value=0)  # Делаем паддинг
    return mfccs, y  # Возвращаем батч с одинаковыми размерами


# Простейшая акустическая модель
class AcousticModel(nn.Module):
    def __init__(self, input_dim=N_MFCC, hidden_dim=128, output_dim=N_MFCC):
        super(AcousticModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


# Загрузка данных
transcriptions = load_transcriptions(tts.xlsx_path)
dataset = TTSDataset("data/prod/audio", transcriptions)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Инициализация модели и оптимизатора
model = AcousticModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Обучение модели
for epoch in range(EPOCHS):
    total_loss = 0
    for mfccs, _ in dataloader:
        optimizer.zero_grad()
        outputs = model(mfccs)  # Прогоняем через модель
        loss = criterion(outputs, mfccs)  # Считаем ошибку
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader)}")

# Сохранение модели
torch.save(model.state_dict(), "tts_acoustic_model.pth")
