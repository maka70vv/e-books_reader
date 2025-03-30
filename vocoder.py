import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import model
import tts


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

# Обучение вокодера
def train_vocoder(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for mfccs, audios in dataloader:
        optimizer.zero_grad()
        mfccs = mfccs.transpose(1, 2)  # Размерность [batch_size, input_dim, seq_len]
        outputs = model(mfccs)  # Восстанавливаем аудио из MFCC
        loss = criterion(outputs, audios)  # Считаем ошибку
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Инициализация модели и оптимизатора
vocoder = Vocoder(input_dim=model.N_MFCC)
criterion = nn.MSELoss()
optimizer = optim.Adam(vocoder.parameters(), lr=model.LEARNING_RATE)

# Загрузка данных
transcriptions = model.load_transcriptions(tts.xlsx_path)
dataset = model.TTSDataset("data/test/normal_voices", transcriptions)
dataloader = DataLoader(dataset, batch_size=model.BATCH_SIZE, shuffle=True)

# Тренировка вокодера
for epoch in range(model.EPOCHS):
    avg_loss = train_vocoder(vocoder, dataloader, criterion, optimizer)
    print(f"Epoch {epoch+1}/{model.EPOCHS}, Loss: {avg_loss}")

# Сохранение модели
torch.save(vocoder.state_dict(), "vocoder.pth")

# Восстановление аудио (пример)
def recover_audio(mfcc, model, sample_rate=model.SAMPLE_RATE):
    mfcc = mfcc.unsqueeze(0).transpose(1, 2)  # Добавляем батч размерности
    with torch.no_grad():
        audio = model(mfcc).squeeze(0).cpu().numpy()  # Восстанавливаем аудио
    return audio

# Пример восстановления аудио
mel_spec_example = torch.tensor(dataset[0][0])  # Берем первое значение
recovered_audio = recover_audio(mel_spec_example, vocoder)

# Сохранение восстановленного аудио
librosa.output.write_wav("recovered_audio.wav", recovered_audio, sr=model.SAMPLE_RATE)  # Для версии librosa >= 0.9 используйте librosa.write_wav