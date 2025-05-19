import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tts
from model import load_transcriptions, TTSDataset, BATCH_SIZE, collate_fn, LEARNING_RATE, AcousticModel, EPOCHS

# Загрузка данных
transcriptions = load_transcriptions(tts.xlsx_path)
dataset = TTSDataset("data/prod/audio", transcriptions)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Инициализация модели и оптимизатора
model = AcousticModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

torch.save(model.state_dict(), "tts_acoustic_model.pth")
