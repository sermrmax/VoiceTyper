from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard


# ============================================================
# НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ============================================================

# Частота записи звука.
# Whisper работает с аудио 16 000 Гц.
SAMPLE_RATE = 16_000

# Один канал — моно.
CHANNELS = 1

# Размер Whisper-модели:
# tiny  — быстрее, но менее точно;
# base  — нормальный вариант для начала;
# small — точнее, но медленнее.
MODEL_SIZE = "base"

# Язык распознавания.
LANGUAGE = "ru"

# Минимальная продолжительность записи.
MINIMUM_RECORDING_SECONDS = 0.3

# Глобальная горячая клавиша.
HOTKEY = "<f8>"


class VoiceTyperApp:
    def __init__(self) -> None:
        # ====================================================
        # ГЛАВНОЕ ОКНО
        # ====================================================

        self.root = tk.Tk()
        self.root.title("Voice Typer")
        self.root.geometry("460x430")
        self.root.resizable(False, False)
        self.root.configure(bg="#15171c")

        # Окно будет отображаться поверх других окон.
        # При необходимости эту строку можно удалить.
        self.root.attributes("-topmost", True)

        # ====================================================
        # СОСТОЯНИЕ ПРИЛОЖЕНИЯ
        # ====================================================

        # Идёт ли сейчас запись.
        self.is_recording = False

        # Выполняется ли распознавание.
        self.is_processing = False

        # Закрывается ли приложение.
        self.is_closing = False

        # Нужно ли после распознавания вставить текст
        # в активное приложение.
        self.auto_paste_after_recognition = False

        # Время последнего нажатия F8.
        # Используется для защиты от двойного срабатывания.
        self.last_hotkey_time = 0.0

        # ====================================================
        # МИКРОФОН
        # ====================================================

        # Поток записи микрофона.
        self.stream: Optional[sd.InputStream] = None

        # Здесь будут находиться части записанного звука.
        self.audio_chunks: list[np.ndarray] = []

        # Блокировка нужна, потому что звук записывается
        # в отдельном системном потоке.
        self.audio_lock = threading.Lock()

        # ====================================================
        # WHISPER
        # ====================================================

        # Сначала модели нет.
        # Она будет загружена в отдельном потоке.
        self.model: Optional[WhisperModel] = None

        # ====================================================
        # КЛАВИАТУРА
        # ====================================================

        # Контроллер для программного нажатия Ctrl+V.
        self.keyboard_controller = keyboard.Controller()

        # Слушатель глобальной горячей клавиши.
        self.hotkey_listener: Optional[
            keyboard.GlobalHotKeys
        ] = None

        # ====================================================
        # ОЧЕРЕДЬ СОБЫТИЙ
        # ====================================================

        # Tkinter нельзя безопасно изменять напрямую
        # из фоновых потоков.
        #
        # Поэтому фоновые потоки помещают события в очередь,
        # а главное окно периодически читает её.
        self.event_queue: queue.Queue[
            tuple[str, object]
        ] = queue.Queue()

        # Создаём интерфейс.
        self.create_interface()

        # Запускаем глобальную клавишу.
        self.start_hotkey_listener()

        # Начинаем проверять очередь событий.
        self.root.after(
            50,
            self.process_events,
        )

        # Назначаем функцию корректного закрытия.
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_app,
        )

        # Загружаем Whisper в отдельном потоке,
        # чтобы интерфейс не завис.
        model_thread = threading.Thread(
            target=self.load_model,
            daemon=True,
        )
        model_thread.start()

    # ========================================================
    # ИНТЕРФЕЙС
    # ========================================================

    def create_interface(self) -> None:
        """Создаёт элементы интерфейса."""

        self.title_label = tk.Label(
            self.root,
            text="VOICE TYPER",
            font=("Segoe UI", 18, "bold"),
            bg="#15171c",
            fg="#ffffff",
        )
        self.title_label.pack(
            pady=(25, 5),
        )

        self.subtitle_label = tk.Label(
            self.root,
            text="Голосовой ввод текста",
            font=("Segoe UI", 10),
            bg="#15171c",
            fg="#777d89",
        )
        self.subtitle_label.pack(
            pady=(0, 12),
        )

        self.status_label = tk.Label(
            self.root,
            text="Загрузка модели...",
            font=("Segoe UI", 11),
            bg="#15171c",
            fg="#f0c75e",
        )
        self.status_label.pack(
            pady=(0, 18),
        )

        self.record_button = tk.Button(
            self.root,
            text="Начать запись",
            command=self.toggle_recording_from_button,

            # Кнопка недоступна, пока модель загружается.
            state=tk.DISABLED,

            font=("Segoe UI", 12, "bold"),

            bg="#3478f6",
            fg="#ffffff",

            activebackground="#2864d7",
            activeforeground="#ffffff",

            disabledforeground="#777b85",

            relief="flat",
            borderwidth=0,
            cursor="hand2",

            width=20,
            height=2,
        )
        self.record_button.pack()

        self.hotkey_label = tk.Label(
            self.root,
            text="F8 — начать или остановить запись",
            font=("Segoe UI", 9),
            bg="#15171c",
            fg="#777d89",
        )
        self.hotkey_label.pack(
            pady=(10, 20),
        )

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
            pady=(0, 5),
        )

        self.result_text = tk.Text(
            self.root,
            height=7,
            width=46,

            font=("Segoe UI", 11),

            bg="#202329",
            fg="#ffffff",

            insertbackground="#ffffff",
            selectbackground="#3478f6",

            relief="flat",
            borderwidth=0,

            wrap="word",

            padx=12,
            pady=12,
        )
        self.result_text.pack(
            padx=30,
        )

    # ========================================================
    # ОЧЕРЕДЬ СОБЫТИЙ
    # ========================================================

    def process_events(self) -> None:
        """
        Обрабатывает события, отправленные
        из фоновых потоков.
        """

        if self.is_closing:
            return

        while True:
            try:
                event_name, event_data = (
                    self.event_queue.get_nowait()
                )
            except queue.Empty:
                break

            if event_name == "model_loaded":
                self.model_loaded()

            elif event_name == "model_error":
                self.show_model_error(
                    str(event_data),
                )

            elif event_name == "transcription_finished":
                self.show_result(
                    str(event_data),
                )

            elif event_name == "transcription_error":
                self.show_transcription_error(
                    str(event_data),
                )

            elif event_name == "hotkey_pressed":
                self.toggle_recording_from_hotkey()

        # Через 50 миллисекунд снова проверяем очередь.
        self.root.after(
            50,
            self.process_events,
        )

    # ========================================================
    # ЗАГРУЗКА МОДЕЛИ
    # ========================================================

    def load_model(self) -> None:
        """Загружает Whisper-модель."""

        try:
            print("Загрузка модели Whisper...")

            model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )

            self.model = model

            print("Модель Whisper загружена")

            self.event_queue.put(
                ("model_loaded", None)
            )

        except Exception as error:
            self.event_queue.put(
                ("model_error", str(error))
            )

    def model_loaded(self) -> None:
        """Включает приложение после загрузки модели."""

        self.status_label.configure(
            text="Готов к записи",
            fg="#72d98b",
        )

        self.record_button.configure(
            state=tk.NORMAL,
        )

    def show_model_error(
        self,
        error_text: str,
    ) -> None:
        """Показывает ошибку загрузки модели."""

        self.status_label.configure(
            text="Ошибка загрузки модели",
            fg="#ff7777",
        )

        messagebox.showerror(
            "Ошибка модели",
            (
                "Не удалось загрузить модель "
                "распознавания.\n\n"
                f"{error_text}"
            ),
        )

    # ========================================================
    # ГЛОБАЛЬНАЯ КЛАВИША F8
    # ========================================================

    def start_hotkey_listener(self) -> None:
        """Запускает глобальную клавишу F8."""

        self.hotkey_listener = keyboard.GlobalHotKeys(
            {
                HOTKEY: self.hotkey_callback,
            }
        )

        self.hotkey_listener.start()

    def hotkey_callback(self) -> None:
        """
        Вызывается библиотекой pynput
        при нажатии F8.

        Функция выполняется не в потоке Tkinter,
        поэтому отправляем событие в очередь.
        """

        current_time = time.monotonic()

        # Защита от слишком быстрого повторного срабатывания.
        if current_time - self.last_hotkey_time < 0.4:
            return

        self.last_hotkey_time = current_time

        self.event_queue.put(
            ("hotkey_pressed", None)
        )

    # ========================================================
    # ЗАПИСЬ МИКРОФОНА
    # ========================================================

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Получает небольшие части аудио
        во время записи.
        """

        if status:
            print("Статус микрофона:", status)

        with self.audio_lock:
            self.audio_chunks.append(
                indata.copy()
            )

    def start_recording(self) -> None:
        """Начинает запись с микрофона."""

        if self.model is None:
            return

        if self.is_processing:
            return

        try:
            # Очищаем предыдущую запись.
            with self.audio_lock:
                self.audio_chunks.clear()

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

            print("Запись началась")

        except Exception as error:
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
        """
        Останавливает запись
        и запускает распознавание.
        """

        if self.stream is None:
            return

        try:
            self.stream.stop()
            self.stream.close()

        except Exception as error:
            print(
                "Ошибка при остановке микрофона:",
                error,
            )

        finally:
            self.stream = None
            self.is_recording = False

        self.record_button.configure(
            text="Начать запись",
            bg="#3478f6",
            activebackground="#2864d7",
            state=tk.DISABLED,
        )

        with self.audio_lock:
            if not self.audio_chunks:
                audio = np.empty(
                    0,
                    dtype=np.float32,
                )
            else:
                audio = np.concatenate(
                    self.audio_chunks,
                    axis=0,
                )

            self.audio_chunks.clear()

        # Переводим массив из формы:
        # [[0.1], [0.2], [0.3]]
        #
        # в форму:
        # [0.1, 0.2, 0.3]
        audio = audio.reshape(-1).astype(
            np.float32,
            copy=False,
        )

        minimum_samples = int(
            SAMPLE_RATE * MINIMUM_RECORDING_SECONDS
        )

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

        transcription_thread = threading.Thread(
            target=self.transcribe_audio,
            args=(audio,),
            daemon=True,
        )
        transcription_thread.start()

    # ========================================================
    # РАСПОЗНАВАНИЕ
    # ========================================================

    def transcribe_audio(
        self,
        audio: np.ndarray,
    ) -> None:
        """Преобразует записанный звук в текст."""

        try:
            if self.model is None:
                raise RuntimeError(
                    "Модель ещё не загружена"
                )

            segments, information = (
                self.model.transcribe(
                    audio,
                    language=LANGUAGE,
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
            )

            text_parts: list[str] = []

            for segment in segments:
                clean_text = segment.text.strip()

                if clean_text:
                    text_parts.append(
                        clean_text
                    )

            result = " ".join(
                text_parts
            )

            print("Распознанный текст:", result)

            self.event_queue.put(
                (
                    "transcription_finished",
                    result,
                )
            )

        except Exception as error:
            self.event_queue.put(
                (
                    "transcription_error",
                    str(error),
                )
            )

    def show_result(
        self,
        result: str,
    ) -> None:
        """
        Показывает результат,
        копирует его и при необходимости вставляет.
        """

        self.is_processing = False

        # Удаляем предыдущий результат.
        self.result_text.delete(
            "1.0",
            tk.END,
        )

        if not result:
            self.status_label.configure(
                text="Речь не распознана",
                fg="#ff7777",
            )

            self.record_button.configure(
                state=tk.NORMAL,
            )

            self.auto_paste_after_recognition = False
            return

        # Показываем текст в приложении.
        self.result_text.insert(
            tk.END,
            result,
        )

        # Копируем текст в буфер обмена.
        pyperclip.copy(result)

        if self.auto_paste_after_recognition:
            self.status_label.configure(
                text="Текст распознан. Вставляю...",
                fg="#72d98b",
            )

            # Небольшая задержка перед Ctrl+V.
            self.root.after(
                250,
                self.paste_text,
            )

        else:
            self.status_label.configure(
                text="Текст скопирован",
                fg="#72d98b",
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
        self.auto_paste_after_recognition = False

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
                "Не удалось преобразовать "
                "речь в текст.\n\n"
                f"{error_text}"
            ),
        )

    # ========================================================
    # ВСТАВКА ТЕКСТА
    # ========================================================

    def paste_text(self) -> None:
        """Имитирует нажатие Ctrl+V."""

        try:
            with self.keyboard_controller.pressed(
                keyboard.Key.ctrl
            ):
                self.keyboard_controller.tap(
                    "v"
                )

            self.status_label.configure(
                text="Текст вставлен",
                fg="#72d98b",
            )

        except Exception as error:
            print(
                "Ошибка автоматической вставки:",
                error,
            )

            self.status_label.configure(
                text="Текст скопирован в буфер",
                fg="#f0c75e",
            )

        finally:
            self.auto_paste_after_recognition = False

    # ========================================================
    # УПРАВЛЕНИЕ ЗАПИСЬЮ
    # ========================================================

    def toggle_recording_from_button(self) -> None:
        """
        Управляет записью через кнопку.

        Через кнопку текст только копируется,
        потому что активным становится окно Voice Typer.
        """

        self.auto_paste_after_recognition = False
        self.toggle_recording()

    def toggle_recording_from_hotkey(self) -> None:
        """
        Управляет записью через F8.

        Активное приложение не меняется,
        поэтому после распознавания можно нажать Ctrl+V.
        """

        self.auto_paste_after_recognition = True
        self.toggle_recording()

    def toggle_recording(self) -> None:
        """Начинает или останавливает запись."""

        if self.is_processing:
            return

        if self.model is None:
            return

        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    # ========================================================
    # ЗАКРЫТИЕ
    # ========================================================

    def close_app(self) -> None:
        """Корректно закрывает приложение."""

        self.is_closing = True

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

            self.stream = None

        if self.hotkey_listener is not None:
            self.hotkey_listener.stop()
            self.hotkey_listener = None

        self.root.destroy()

    def run(self) -> None:
        """Запускает приложение."""

        self.root.mainloop()


if __name__ == "__main__":
    app = VoiceTyperApp()
    app.run()