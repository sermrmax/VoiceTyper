from faster_whisper import WhisperModel


AUDIO_FILE = "../recording.wav"
MODEL_SIZE = "base"


print("Загрузка модели...")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
)

print("Распознавание началось...")

segments, info = model.transcribe(
    AUDIO_FILE,
    language="ru",
    beam_size=5,
    vad_filter=True,
)

text_parts = []

for segment in segments:
    text_parts.append(segment.text.strip())

result_text = " ".join(text_parts)

print()
print("Результат:")
print(result_text)