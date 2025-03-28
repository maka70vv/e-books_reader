import tts
from parsers import extract_text_from_pdf
from tts import generate_speech

text_from_pdf = extract_text_from_pdf("input/output.pdf")
print(text_from_pdf)

generate_speech(text_from_pdf, tts.audio_data, "output/output_dtw.wav")
