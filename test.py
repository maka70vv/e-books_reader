import torch
from torch.utils.data import DataLoader

import model
import tts
from model import load_transcriptions, TTSDataset


def test_tts_model():
    # Загрузка данных для тестирования
    test_transcriptions = load_transcriptions(tts.xlsx_path)
    test_dataset = TTSDataset("data/prod/audio", test_transcriptions)
    test_dataloader = DataLoader(test_dataset, batch_size=model.BATCH_SIZE, shuffle=False)

    # Инициализация модели
    model_obj = model.AcousticModel()  # Создаем экземпляр модели
    model_obj.load_state_dict(torch.load("tts_acoustic_model.pth"))  # Загрузка сохраненной модели
    model_obj.eval()  # Устанавливаем модель в режим тестирования

    # Прогонка данных через модель
    total_loss = 0
    with torch.no_grad():  # Отключаем градиенты
        for mfccs, _ in test_dataloader:
            # Прогоняем через модель
            outputs = model_obj(mfccs)
            # Рассчитываем ошибку
            loss = model.criterion(outputs, mfccs)
            total_loss += loss.item()

    # Средняя ошибка на тестовом наборе
    print(f"Test Loss: {total_loss / len(test_dataloader)}")


test_tts_model()