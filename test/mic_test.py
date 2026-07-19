import sounddevice as sd
import soundfile as sf


DURATION = 5
SAMPLE_RATE = 16_000
CHANNELS = 1
OUTPUT_FILE = "test_recording.wav"


print("Запись началась. Говори...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
)

sd.wait()

sf.write(
    OUTPUT_FILE,
    audio,
    SAMPLE_RATE,
)

print(f"Запись завершена: {OUTPUT_FILE}")