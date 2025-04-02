import os

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

    # Загружаем аудиофайлы и их MFCC
    transcriptions = {}
    for _, row in df.iterrows():
        wav_file = row["file"]  # Используем значение из столбца "file" для аудиофайла
        text = row["text"]  # Текст из столбца "text"
        text = filter_text(text)  # Применяем фильтрацию текста
        audio_path = os.path.join(tts.dataset_path, f"{wav_file}.wav")  # Создаем путь к аудиофайлу

        if os.path.exists(audio_path):
            y, sr = librosa.load(audio_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            transcriptions[wav_file] = (text, mfcc)  # Сохраняем имя файла и текст
    return transcriptions


# Подготовка данных
class TTSDataset(Dataset):
    def __init__(self, dataset_path, transcriptions, max_length=150):
        self.dataset_path = dataset_path
        self.transcriptions = transcriptions
        self.audio_files = list(transcriptions.keys())
        self.max_length = max_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_name = self.audio_files[idx]
        text, _ = self.transcriptions[file_name]
        audio_path = os.path.join(self.dataset_path, f"{file_name}.wav")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        if mfcc.shape[1] > self.max_length:
            mfcc = mfcc[:, :self.max_length]  # Обрезаем
        elif mfcc.shape[1] < self.max_length:
            padding = self.max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')  # Паддинг

        # Транспонируем MFCC для соответствия входному размеру LSTM
        mfcc = mfcc.T  # Теперь размерность будет [seq_len, input_size] (150, 40)

        return torch.tensor(mfcc, dtype=torch.float32), y  # Возвращаем аудио также для вокодера



# Простейшая акустическая модель
class AcousticModel(nn.Module):
    def __init__(self, input_dim=N_MFCC, hidden_dim=128, output_dim=N_MFCC):
        super(AcousticModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x имеет размерность [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


# Загрузка данных
transcriptions = load_transcriptions(tts.xlsx_path)
dataset = TTSDataset("data/prod/audio", transcriptions)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader)}")

# Сохранение модели
torch.save(model.state_dict(), "tts_acoustic_model.pth")
