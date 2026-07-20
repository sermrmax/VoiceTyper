from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


# Настройки записи
SAMPLE_RATE = 16_000
CHANNELS = 1

# Настройки распознавания
MODEL_SIZE = "base"
LANGUAGE = "ru"


class VoiceTyperApp:
    def __init__(self) -> None:
        # -----------------------------
        # Главное окно
        # -----------------------------
        self.root = tk.Tk()
        self.root.title("Voice Typer")
        self.root.geometry("440x400")
        self.root.resizable(False, False)
        self.root.configure(bg="#15171c")

        # -----------------------------
        # Состояние приложения
        # -----------------------------
        self.is_recording = False
        self.is_processing = False
        self.is_closing = False

        # Поток микрофона
        self.stream: Optional[sd.InputStream] = None

        # Список фрагментов записанного звука
        self.audio_chunks: list[np.ndarray] = []

        # Модель Whisper
        self.model: Optional[WhisperModel] = None

        # Создаём интерфейс
        self.create_interface()

        # Обработчик закрытия окна
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_app,
        )

        # Загружаем модель в отдельном потоке,
        # чтобы окно приложения не зависало
        model_thread = threading.Thread(
            target=self.load_model,
            daemon=True,
        )
        model_thread.start()

    def create_interface(self) -> None:
        """Создаёт элементы интерфейса."""

        # Заголовок приложения
        self.title_label = tk.Label(
            self.root,
            text="VOICE TYPER",
            font=("Segoe UI", 18, "bold"),
            bg="#15171c",
            fg="white",
        )
        self.title_label.pack(
            pady=(25, 8),
        )

        # Подзаголовок
        self.subtitle_label = tk.Label(
            self.root,
            text="Преобразование голоса в текст",
            font=("Segoe UI", 10),
            bg="#15171c",
            fg="#737985",
        )
        self.subtitle_label.pack(
            pady=(0, 10),
        )

        # Статус приложения
        self.status_label = tk.Label(
            self.root,
            text="Загрузка модели...",
            font=("Segoe UI", 11),
            bg="#15171c",
            fg="#f0c75e",
        )
        self.status_label.pack(
            pady=(0, 20),
        )

        # Кнопка записи
        self.record_button = tk.Button(
            self.root,
            text="Начать запись",
            command=self.toggle_recording,

            # Пока модель загружается,
            # кнопка выключена
            state=tk.DISABLED,

            font=("Segoe UI", 12, "bold"),
            bg="#3478f6",
            fg="white",
            activebackground="#2864d7",
            activeforeground="white",
            disabledforeground="#777b85",

            relief="flat",
            borderwidth=0,
            cursor="hand2",
            width=18,
            height=2,
        )
        self.record_button.pack()

        # Подпись над текстовым полем
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

        # Поле для распознанного текста
        self.result_text = tk.Text(
            self.root,
            height=6,
            width=44,
            font=("Segoe UI", 11),
            bg="#202329",
            fg="white",
            insertbackground="white",
            selectbackground="#3478f6",
            relief="flat",
            borderwidth=0,
            wrap="word",
            padx=10,
            pady=10,
        )
        self.result_text.pack(
            padx=30,
        )

    def load_model(self) -> None:
        """Загружает модель распознавания речи."""

        try:
            print("Загрузка модели Whisper...")

            self.model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )

            print("Модель Whisper загружена")

            # Интерфейс Tkinter нужно изменять
            # только из главного потока
            if not self.is_closing:
                self.root.after(
                    0,
                    self.model_loaded,
                )

        except Exception as error:
            error_text = str(error)

            if not self.is_closing:
                self.root.after(
                    0,
                    lambda: self.show_model_error(error_text),
                )

    def model_loaded(self) -> None:
        """Включает кнопку после загрузки модели."""

        self.status_label.configure(
            text="Готов к записи",
            fg="#72d98b",
        )

        self.record_button.configure(
            state=tk.NORMAL,
        )

    def show_model_error(self, error_text: str) -> None:
        """Показывает ошибку загрузки Whisper."""

        self.status_label.configure(
            text="Ошибка загрузки модели",
            fg="#ff7777",
        )

        messagebox.showerror(
            "Ошибка модели",
            (
                "Не удалось загрузить модель распознавания.\n\n"
                f"{error_text}"
            ),
        )

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Получает небольшие фрагменты аудио.

        Эта функция автоматически вызывается
        библиотекой sounddevice во время записи.
        """

        if status:
            print("Статус микрофона:", status)

        # Обязательно создаём копию.
        # Иначе sounddevice может перезаписать данные.
        self.audio_chunks.append(
            indata.copy(),
        )

    def start_recording(self) -> None:
        """Начинает запись с микрофона."""

        if self.model is None:
            return

        if self.is_processing:
            return

        try:
            # Очищаем предыдущую запись
            self.audio_chunks.clear()

            # Создаём входной аудиопоток
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self.audio_callback,
            )

            # Запускаем микрофон
            self.stream.start()

            self.is_recording = True

            # Меняем внешний вид кнопки
            self.record_button.configure(
                text="Остановить запись",
                bg="#d94a4a",
                activebackground="#b83b3b",
            )

            self.status_label.configure(
                text="Идёт запись...",
                fg="#ff7777",
            )

            print("Запись началась")

        except Exception as error:
            # Если поток успел создаться,
            # пытаемся корректно его закрыть
            if self.stream is not None:
                try:
                    self.stream.close()
                except Exception:
                    pass

                self.stream = None

            self.is_recording = False

            messagebox.showerror(
                "Ошибка микрофона",
                (
                    "Не удалось начать запись.\n\n"
                    f"{error}"
                ),
            )

    def stop_recording(self) -> None:
        """Останавливает запись и запускает распознавание."""

        if self.stream is None:
            return

        try:
            self.stream.stop()
            self.stream.close()

        except Exception as error:
            print("Ошибка остановки микрофона:", error)

        finally:
            self.stream = None
            self.is_recording = False

        # Возвращаем обычный вид кнопки
        self.record_button.configure(
            text="Начать запись",
            bg="#3478f6",
            activebackground="#2864d7",
            state=tk.DISABLED,
        )

        # Проверяем, записался ли звук
        if not self.audio_chunks:
            self.status_label.configure(
                text="Звук не был записан",
                fg="#ff7777",
            )

            self.record_button.configure(
                state=tk.NORMAL,
            )
            return

        # Объединяем все части звука
        # в один большой NumPy-массив
        audio = np.concatenate(
            self.audio_chunks,
            axis=0,
        )

        # Очищаем фрагменты после объединения
        self.audio_chunks.clear()

        # Whisper ожидает одномерный массив:
        # [0.1, 0.2, 0.3, ...]
        audio = audio.reshape(-1)

        # Гарантируем формат float32
        audio = audio.astype(
            np.float32,
            copy=False,
        )

        # Проверяем, что запись не слишком короткая
        minimum_samples = SAMPLE_RATE // 4

        if audio.size < minimum_samples:
            self.status_label.configure(
                text="Запись слишком короткая",
                fg="#ff7777",
            )

            self.record_button.configure(
                state=tk.NORMAL,
            )
            return

        self.is_processing = True

        self.status_label.configure(
            text="Распознавание...",
            fg="#f0c75e",
        )

        print("Запись остановлена")
        print("Началось распознавание")

        # Распознавание может занимать несколько секунд.
        # Поэтому запускаем его в отдельном потоке.
        transcription_thread = threading.Thread(
            target=self.transcribe_audio,
            args=(audio,),
            daemon=True,
        )
        transcription_thread.start()

    def transcribe_audio(
        self,
        audio: np.ndarray,
    ) -> None:
        """Преобразует аудиомассив в текст."""

        try:
            if self.model is None:
                raise RuntimeError(
                    "Модель распознавания не загружена"
                )

            segments, info = self.model.transcribe(
                audio,

                # Русский язык
                language=LANGUAGE,

                # Качество поиска результата
                beam_size=5,

                # Убирает участки тишины
                vad_filter=True,

                # Не связывает текущую запись
                # с предыдущими фрагментами
                condition_on_previous_text=False,
            )

            text_parts: list[str] = []

            # faster-whisper возвращает текст частями
            for segment in segments:
                clean_text = segment.text.strip()

                if clean_text:
                    text_parts.append(clean_text)

            # Соединяем сегменты через пробел
            result = " ".join(text_parts)

            print("Результат:", result)

            if not self.is_closing:
                self.root.after(
                    0,
                    lambda: self.show_result(result),
                )

        except Exception as error:
            error_text = str(error)

            if not self.is_closing:
                self.root.after(
                    0,
                    lambda: self.show_transcription_error(
                        error_text
                    ),
                )

    def show_result(self, result: str) -> None:
        """Выводит распознанный текст в интерфейс."""

        self.is_processing = False

        # Удаляем старый текст
        self.result_text.delete(
            "1.0",
            tk.END,
        )

        if result:
            # Выводим новый текст
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

        # Снова разрешаем запись
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
            (
                "Не удалось преобразовать речь в текст.\n\n"
                f"{error_text}"
            ),
        )

    def toggle_recording(self) -> None:
        """Запускает или останавливает запись."""

        # Пока идёт распознавание,
        # новые нажатия игнорируются
        if self.is_processing:
            return

        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def close_app(self) -> None:
        """Корректно закрывает приложение."""

        self.is_closing = True

        # Закрываем микрофон, если запись ещё идёт
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

            self.stream = None

        self.root.destroy()

    def run(self) -> None:
        """Запускает главный цикл Tkinter."""

        self.root.mainloop()


if __name__ == "__main__":
    app = VoiceTyperApp()
    app.run()