import tts
from parsers import extract_text_from_pdf, extract_text_from_fb2
from tts import generate_speech
#
text_from_pdf = extract_text_from_pdf("input/skazka-o-rybake-i-rybke.pdf")
#
# text_from_epub = extract_text_from_pdf("input/input_epub.epub")
# print(text_from_epub)
#
# text_from_fb2 = extract_text_from_fb2("input/input_fb2.fb2")
# print(text_from_fb2)
#
generate_speech(text_from_pdf, tts.audio_data, "output/output_pushkin.wav")
# generate_speech(text_from_epub, tts.audio_data, "output/output_dtw_epub.wav")
# generate_speech(text_from_fb2, tts.audio_data, "output/output_dtw_fb2.wav")
import torch

from model import AcousticModel


def load_model():
    # Загрузка модели
    model = AcousticModel()
    model.load_state_dict(torch.load("tts_acoustic_model.pth"))
    model.eval()  # Устанавливаем модель в режим тестирования
