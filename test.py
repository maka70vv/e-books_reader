import torch
from torch.utils.data import DataLoader

import model
import tts
from model import load_transcriptions, TTSDataset


def test_tts_model():
    # Загрузка данных для тестирования (используем ту же функцию load_transcriptions)
    test_transcriptions = load_transcriptions(tts.xlsx_path)
    test_dataset = TTSDataset("data/test/normal_voices", test_transcriptions)
    test_dataloader = DataLoader(test_dataset, batch_size=model.BATCH_SIZE, shuffle=False)

    # Прогонка данных через модель
    total_loss = 0
    with torch.no_grad():  # Отключаем градиенты
        model.eval()  # Устанавливаем модель в режим тестирования
        for mfccs, _ in test_dataloader:
            # Прогоняем через модель
            outputs = model(mfccs)
            # Рассчитываем ошибку
            loss = model.criterion(outputs, mfccs)
            total_loss += loss.item()

    # Средняя ошибка на тестовом наборе
    print(f"Test Loss: {total_loss / len(test_dataloader)}")
