import threading
import tkinter as tk
from tkinter import messagebox

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16_000
CHANNELS = 1
MODEL_SIZE = "base"
LANGUAGE = "ru"


class VoiceTyperApp:
    def __init__(self) -> None:
        self.root = tk.Tk()

        self.root.title("Voice Typer")
        self.root.geometry("440x390")
        self.root.resizable(False, False)
        self.root.configure(bg="#15171c")

        # Состояние приложения
        self.is_recording = False
        self.is_processing = False

        # Работа с микрофоном
        self.stream = None
        self.audio_chunks = []

        # Модель распознавания
        self.model = None

        self.create_interface()

        # Корректное закрытие приложения
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

        # Загружаем модель в отдельном потоке
        threading.Thread(
            target=self.load_model,
            daemon=True,
        ).start()

    def create_interface(self) -> None:
        """Создаёт интерфейс приложения."""

        self.title_label = tk.Label(
            self.root,
            text="VOICE TYPER",
            font=("Segoe UI", 18, "bold"),
            bg="#15171c",
            fg="white",
        )
        self.title_label.pack(pady=(25, 8))

        self.status_label = tk.Label(
            self.root,
            text="Загрузка модели...",
            font=("Segoe UI", 11),
            bg="#15171c",
            fg="#a5a9b3",
        )
        self.status_label.pack(pady=(0, 20))

        self.record_button = tk.Button(
            self.root,
            text="Начать запись",
            command=self.toggle_recording,
            state=tk.DISABLED,
            font=("Segoe UI", 12, "bold"),
            bg="#3478f6",
            fg="white",
            activebackground="#2864d7",
            activeforeground="white",
            disabledbackground="#3a3d44",
            disabledforeground="#777b85",
            relief="flat",
            cursor="hand2",
            width=18,
            height=2,
        )
        self.record_button.pack()

        self.result_label = tk.Label(
            self.root,
            text="Распознанный текст:",
            font=("Segoe UI", 10),
            bg="#15171c",
            fg="#a5a9b3",
        )
        self.result_label.pack(
            anchor="w",
            padx=30,
            pady=(25, 5),
        )

        self.result_text = tk.Text(
            self.root,
            height=6,
            width=44,
            font=("Segoe UI", 11),
            bg="#202329",
            fg="white",
            insertbackground="white",
            relief="flat",
            wrap="word",
            padx=10,
            pady=10,
        )
        self.result_text.pack(padx=30)

    def load_model(self) -> None:
        """Загружает Whisper-модель."""

        try:
            self.model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )

            self.root.after(
                0,
                self.model_loaded,
            )

        except Exception as error:
            error_text = str(error)

            self.root.after(
                0,
                lambda: self.show_model_error(error_text),
            )

    def model_loaded(self) -> None:
        """Вызывается после успешной загрузки модели."""

        self.status_label.configure(
            text="Готов к записи",
            fg="#72d98b",
        )

        self.record_button.configure(
            state=tk.NORMAL,
        )

    def show_model_error(self, error_text: str) -> None:
        """Показывает ошибку загрузки модели."""

        self.status_label.configure(
            text="Ошибка загрузки модели",
            fg="#ff7777",
        )

        messagebox.showerror(
            "Ошибка модели",
            error_text,
        )

    def audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ) -> None:
        """Получает фрагменты звука с микрофона."""

        if status:
            print("Статус микрофона:", status)

        self.audio_chunks.append(indata.copy())

    def start_recording(self) -> None:
        """Начинает запись."""

        if self.model is None:
            return

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
        """Останавливает запись и запускает распознавание."""

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
            state=tk.DISABLED,
        )

        if not self.audio_chunks:
            self.status_label.configure(
                text="Звук не записан",
                fg="#ff7777",
            )

            self.record_button.configure(
                state=tk.NORMAL,
            )
            return

        # Соединяем фрагменты записи
        audio = np.concatenate(
            self.audio_chunks,
            axis=0,
        )

        # Whisper ожидает одномерный массив
        audio = audio.reshape(-1).astype(np.float32)

        self.is_processing = True

        self.status_label.configure(
            text="Распознавание...",
            fg="#f0c75e",
        )

        # Распознавание выполняется отдельно от интерфейса
        threading.Thread(
            target=self.transcribe_audio,
            args=(audio,),
            daemon=True,
        ).start()

    def transcribe_audio(self, audio: np.ndarray) -> None:
        """Преобразует аудио в текст."""

        try:
            segments, info = self.model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            text_parts = []

            for segment in segments:
                clean_text = segment.text.strip()

                if clean_text:
                    text_parts.append(clean_text)

            result = " ".join(text_parts)

            self.root.after(
                0,
                lambda: self.show_result(result),
            )

        except Exception as error:
            error_text = str(error)

            self.root.after(
                0,
                lambda: self.show_transcription_error(error_text),
            )

    def show_result(self, result: str) -> None:
        """Выводит распознанный текст."""

        self.is_processing = False

        self.result_text.delete(
            "1.0",
            tk.END,
        )

        if result:
            self.result_text.insert(
                tk.END,
                result,
            )

            self.status_label.configure(
                text="Текст распознан",
                fg="#72d98b",
            )
        else:
            self.status_label.configure(
                text="Речь не распознана",
                fg="#ff7777",
            )

        self.record_button.configure(
            state=tk.NORMAL,
        )

    def show_transcription_error(
        self,
        error_text: str,
    ) -> None:
        """Показывает ошибку распознавания."""

        self.is_processing = False

        self.status_label.configure(
            text="Ошибка распознавания",
            fg="#ff7777",
        )

        self.record_button.configure(
            state=tk.NORMAL,
        )

        messagebox.showerror(
            "Ошибка распознавания",
            error_text,
        )

    def toggle_recording(self) -> None:
        """Переключает запись."""

        if self.is_processing:
            return

        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def close_app(self) -> None:
        """Закрывает микрофон и приложение."""

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = VoiceTyperApp()
    app.run()