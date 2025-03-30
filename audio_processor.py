import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def extract_mel_spectrogram(audio_path, sample_rate=22050, n_mfcc=40, hop_length=512):
    y, sr = librosa.load(audio_path, sr=sample_rate)
    # Extracting Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mfcc, hop_length=hop_length)
    return mel_spectrogram

# Пример для тестового аудиофайла
audio_path = "data/test/normal_voices/48.wav"
mel_spectrogram = extract_mel_spectrogram(audio_path)

# Визуализируем Mel-спектрограмму
plt.figure(figsize=(10, 6))
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()
