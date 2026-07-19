import tkinter as tk
from tkinter import messagebox

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16_000
CHANNELS = 1
OUTPUT_FILE = "recording.wav"


class VoiceTyperApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Voice Typer")
        self.root.geometry("400x260")
        self.root.resizable(False, False)
        self.root.configure(bg="#15171c")

        # Состояние записи
        self.is_recording = False

        # Здесь будет находиться поток микрофона
        self.stream = None

        # Здесь будут храниться записанные кусочки звука
        self.audio_chunks = []

        self.title_label = tk.Label(
            self.root,
            text="VOICE TYPER",
            font=("Segoe UI", 18, "bold"),
            bg="#15171c",
            fg="white",
        )
        self.title_label.pack(pady=(30, 8))

        self.status_label = tk.Label(
            self.root,
            text="Готов к записи",
            font=("Segoe UI", 11),
            bg="#15171c",
            fg="#a5a9b3",
        )
        self.status_label.pack(pady=(0, 25))

        self.record_button = tk.Button(
            self.root,
            text="Начать запись",
            command=self.toggle_recording,
            font=("Segoe UI", 12, "bold"),
            bg="#3478f6",
            fg="white",
            activebackground="#2864d7",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            width=18,
            height=2,
        )
        self.record_button.pack()

        # Что делать при закрытии окна
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ) -> None:
        """Получает небольшие части звука с микрофона."""

        if status:
            print("Статус микрофона:", status)

        # copy() важен: сохраняем отдельную копию аудиоданных
        self.audio_chunks.append(indata.copy())

    def start_recording(self) -> None:
        """Начинает запись с микрофона."""

        try:
            self.audio_chunks = []

            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self.audio_callback,
            )

            self.stream.start()
            self.is_recording = True

            self.record_button.configure(
                text="Остановить запись",
                bg="#d94a4a",
                activebackground="#b83b3b",
            )

            self.status_label.configure(
                text="Идёт запись...",
                fg="#ff7777",
            )

        except Exception as error:
            messagebox.showerror(
                "Ошибка микрофона",
                str(error),
            )

    def stop_recording(self) -> None:
        """Останавливает запись и сохраняет WAV-файл."""

        if self.stream is None:
            return

        self.stream.stop()
        self.stream.close()
        self.stream = None

        self.is_recording = False

        self.record_button.configure(
            text="Начать запись",
            bg="#3478f6",
            activebackground="#2864d7",
        )

        if not self.audio_chunks:
            self.status_label.configure(
                text="Звук не был записан",
                fg="#a5a9b3",
            )
            return

        # Соединяем все маленькие части записи в один массив
        audio = np.concatenate(self.audio_chunks, axis=0)

        # Сохраняем запись
        sf.write(
            OUTPUT_FILE,
            audio,
            SAMPLE_RATE,
        )

        self.status_label.configure(
            text=f"Сохранено: {OUTPUT_FILE}",
            fg="#72d98b",
        )

        print(f"Аудио сохранено в {OUTPUT_FILE}")

    def toggle_recording(self) -> None:
        """Переключает запись по нажатию кнопки."""

        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def close_app(self) -> None:
        """Корректно закрывает поток микрофона и приложение."""

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = VoiceTyperApp()
    app.run()