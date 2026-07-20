from __future__ import annotations

import ctypes
import queue
import threading
import time
import tkinter as tk
from ctypes import wintypes
from tkinter import messagebox
from typing import Optional

import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard


# ============================================================
# НАСТРОЙКИ
# ============================================================

SAMPLE_RATE = 16_000
CHANNELS = 1

MODEL_SIZE = "base"
LANGUAGE = "ru"

HOTKEY = "<f8>"

MIN_RECORDING_SECONDS = 0.3


# ============================================================
# WINDOWS API
# ============================================================

GA_ROOT = 2
SW_RESTORE = 9

INPUT_KEYBOARD = 1

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_V = 0x56


if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("data",)

    _fields_ = [
        ("type", wintypes.DWORD),
        ("data", INPUT_UNION),
    ]


# ============================================================
# УПРАВЛЕНИЕ ОКНАМИ WINDOWS
# ============================================================

class WindowsController:
    """Запоминает целевое окно и отправляет в него текст."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root

        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32

        self.configure_api()

        self.root.update_idletasks()

        self.app_window = self.get_root_window(
            self.root.winfo_id()
        )

        self.last_external_window = 0
        self.target_window = 0

    def configure_api(self) -> None:
        """Настраивает типы функций Windows API."""

        self.user32.GetForegroundWindow.restype = (
            wintypes.HWND
        )

        self.user32.GetAncestor.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
        ]
        self.user32.GetAncestor.restype = wintypes.HWND

        self.user32.IsWindow.argtypes = [
            wintypes.HWND,
        ]
        self.user32.IsWindow.restype = wintypes.BOOL

        self.user32.ShowWindow.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
        ]
        self.user32.ShowWindow.restype = wintypes.BOOL

        self.user32.BringWindowToTop.argtypes = [
            wintypes.HWND,
        ]
        self.user32.BringWindowToTop.restype = (
            wintypes.BOOL
        )

        self.user32.SetForegroundWindow.argtypes = [
            wintypes.HWND,
        ]
        self.user32.SetForegroundWindow.restype = (
            wintypes.BOOL
        )

        self.user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.user32.GetWindowThreadProcessId.restype = (
            wintypes.DWORD
        )

        self.user32.AttachThreadInput.argtypes = [
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.BOOL,
        ]
        self.user32.AttachThreadInput.restype = (
            wintypes.BOOL
        )

        self.user32.GetWindowTextLengthW.argtypes = [
            wintypes.HWND,
        ]
        self.user32.GetWindowTextLengthW.restype = (
            ctypes.c_int
        )

        self.user32.GetWindowTextW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        self.user32.GetWindowTextW.restype = ctypes.c_int

        self.user32.GetClassNameW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        self.user32.GetClassNameW.restype = ctypes.c_int

        self.user32.SendInput.argtypes = [
            wintypes.UINT,
            ctypes.POINTER(INPUT),
            ctypes.c_int,
        ]
        self.user32.SendInput.restype = wintypes.UINT

        self.kernel32.GetCurrentThreadId.restype = (
            wintypes.DWORD
        )

    def get_root_window(
        self,
        window: int | None,
    ) -> int:
        """Возвращает главное окно приложения."""

        if not window:
            return 0

        root_window = self.user32.GetAncestor(
            wintypes.HWND(window),
            GA_ROOT,
        )

        if root_window:
            return int(root_window)

        return int(window)

    def is_valid_window(self, window: int) -> bool:
        """Проверяет существование окна."""

        if not window:
            return False

        return bool(
            self.user32.IsWindow(
                wintypes.HWND(window)
            )
        )

    def get_foreground_window(self) -> int:
        """Возвращает активное окно."""

        raw_window = (
            self.user32.GetForegroundWindow()
        )

        # Windows иногда возвращает NULL.
        # В Python это будет None.
        if not raw_window:
            return 0

        return self.get_root_window(
            int(raw_window)
        )

    def update_last_external_window(self) -> None:
        """Запоминает последнее внешнее окно."""

        window = self.get_foreground_window()

        if not self.is_valid_window(window):
            return

        if window == self.app_window:
            return

        self.last_external_window = window

    def select_target_window(
        self,
        captured_window: int,
    ) -> bool:
        """Выбирает окно для будущей вставки."""

        candidate = self.get_root_window(
            captured_window
        )

        if (
            self.is_valid_window(candidate)
            and candidate != self.app_window
        ):
            self.target_window = candidate
            self.last_external_window = candidate

        elif self.is_valid_window(
            self.last_external_window
        ):
            self.target_window = (
                self.last_external_window
            )

        else:
            self.target_window = 0
            return False

        print(
            "Целевое окно:",
            self.get_window_title(
                self.target_window
            ),
        )

        return True

    def get_window_title(
        self,
        window: int | None = None,
    ) -> str:
        """Возвращает заголовок окна."""

        hwnd = window or self.target_window

        if not self.is_valid_window(hwnd):
            return ""

        length = self.user32.GetWindowTextLengthW(
            wintypes.HWND(hwnd)
        )

        buffer = ctypes.create_unicode_buffer(
            max(length + 1, 2)
        )

        self.user32.GetWindowTextW(
            wintypes.HWND(hwnd),
            buffer,
            len(buffer),
        )

        return buffer.value

    def get_window_class(
        self,
        window: int | None = None,
    ) -> str:
        """Возвращает системный класс окна."""

        hwnd = window or self.target_window

        if not self.is_valid_window(hwnd):
            return ""

        buffer = ctypes.create_unicode_buffer(256)

        self.user32.GetClassNameW(
            wintypes.HWND(hwnd),
            buffer,
            len(buffer),
        )

        return buffer.value

    def get_window_thread_id(
        self,
        window: int,
    ) -> int:
        """Возвращает ID потока окна."""

        if not self.is_valid_window(window):
            return 0

        process_id = wintypes.DWORD(0)

        thread_id = (
            self.user32.GetWindowThreadProcessId(
                wintypes.HWND(window),
                ctypes.byref(process_id),
            )
        )

        return int(thread_id)

    def is_target_foreground(self) -> bool:
        """Проверяет, активно ли целевое окно."""

        if not self.is_valid_window(
            self.target_window
        ):
            return False

        return (
            self.get_foreground_window()
            == self.target_window
        )

    def restore_target_window(self) -> bool:
        """Возвращает целевое окно на передний план."""

        target = self.target_window

        if not self.is_valid_window(target):
            self.target_window = 0
            return False

        current_window = (
            self.get_foreground_window()
        )

        if current_window == target:
            return True

        current_thread = int(
            self.kernel32.GetCurrentThreadId()
        )

        target_thread = (
            self.get_window_thread_id(target)
        )

        foreground_thread = (
            self.get_window_thread_id(
                current_window
            )
        )

        attached_threads: list[int] = []

        try:
            for thread_id in {
                target_thread,
                foreground_thread,
            }:
                if (
                    thread_id
                    and thread_id != current_thread
                ):
                    attached = (
                        self.user32.AttachThreadInput(
                            current_thread,
                            thread_id,
                            True,
                        )
                    )

                    if attached:
                        attached_threads.append(
                            thread_id
                        )

            self.user32.ShowWindow(
                wintypes.HWND(target),
                SW_RESTORE,
            )

            self.user32.BringWindowToTop(
                wintypes.HWND(target)
            )

            self.user32.SetForegroundWindow(
                wintypes.HWND(target)
            )

        finally:
            for thread_id in reversed(
                attached_threads
            ):
                self.user32.AttachThreadInput(
                    current_thread,
                    thread_id,
                    False,
                )

        time.sleep(0.15)

        return self.is_target_foreground()

    def is_terminal(self) -> bool:
        """Проверяет, является ли окно терминалом."""

        window_info = (
            self.get_window_title().lower()
            + " "
            + self.get_window_class().lower()
        )

        terminal_markers = (
            "windows terminal",
            "powershell",
            "command prompt",
            "командная строка",
            "consolewindowclass",
            "cascadia_hosting_window_class",
        )

        return any(
            marker in window_info
            for marker in terminal_markers
        )

    @staticmethod
    def create_virtual_key_input(
        virtual_key: int,
        key_up: bool = False,
    ) -> INPUT:
        """Создаёт событие обычной клавиши."""

        event = INPUT()
        event.type = INPUT_KEYBOARD

        event.ki = KEYBDINPUT(
            wVk=virtual_key,
            wScan=0,
            dwFlags=(
                KEYEVENTF_KEYUP
                if key_up
                else 0
            ),
            time=0,
            dwExtraInfo=0,
        )

        return event

    @staticmethod
    def create_unicode_input(
        code_unit: int,
        key_up: bool = False,
    ) -> INPUT:
        """Создаёт событие Unicode-символа."""

        event = INPUT()
        event.type = INPUT_KEYBOARD

        flags = KEYEVENTF_UNICODE

        if key_up:
            flags |= KEYEVENTF_KEYUP

        event.ki = KEYBDINPUT(
            wVk=0,
            wScan=code_unit,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )

        return event

    def send_inputs(
        self,
        events: list[INPUT],
    ) -> bool:
        """Отправляет список клавиатурных событий."""

        if not events:
            return False

        array_type = INPUT * len(events)
        input_array = array_type(*events)

        sent_count = self.user32.SendInput(
            len(events),
            input_array,
            ctypes.sizeof(INPUT),
        )

        return int(sent_count) == len(events)

    def send_paste_shortcut(self) -> bool:
        """Отправляет Ctrl+V или Ctrl+Shift+V."""

        if not self.is_target_foreground():
            return False

        modifiers = [VK_CONTROL]

        if self.is_terminal():
            modifiers.append(VK_SHIFT)

        events: list[INPUT] = []

        # Нажимаем модификаторы.
        for virtual_key in modifiers:
            events.append(
                self.create_virtual_key_input(
                    virtual_key
                )
            )

        # Нажимаем V.
        events.append(
            self.create_virtual_key_input(VK_V)
        )

        # Отпускаем V.
        events.append(
            self.create_virtual_key_input(
                VK_V,
                key_up=True,
            )
        )

        # Отпускаем модификаторы.
        for virtual_key in reversed(
            modifiers
        ):
            events.append(
                self.create_virtual_key_input(
                    virtual_key,
                    key_up=True,
                )
            )

        return self.send_inputs(events)

    def send_unicode_text(
        self,
        text: str,
    ) -> bool:
        """Печатает текст напрямую."""

        if not text:
            return False

        if not self.is_target_foreground():
            return False

        encoded_text = text.encode(
            "utf-16-le"
        )

        events: list[INPUT] = []

        for index in range(
            0,
            len(encoded_text),
            2,
        ):
            code_unit = int.from_bytes(
                encoded_text[
                    index:index + 2
                ],
                byteorder="little",
            )

            events.append(
                self.create_unicode_input(
                    code_unit
                )
            )

            events.append(
                self.create_unicode_input(
                    code_unit,
                    key_up=True,
                )
            )

        batch_size = 160

        for index in range(
            0,
            len(events),
            batch_size,
        ):
            batch = events[
                index:index + batch_size
            ]

            if not self.send_inputs(batch):
                return False

            time.sleep(0.01)

        return True


# ============================================================
# ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ============================================================

class VoiceTyperApp:
    def __init__(self) -> None:
        self.root = tk.Tk()

        self.root.title("Voice Typer")
        self.root.geometry("460x430")
        self.root.resizable(False, False)
        self.root.configure(bg="#15171c")

        # Окно не должно быть поверх всех окон.
        self.root.attributes(
            "-topmost",
            False,
        )

        # Состояние приложения.
        self.is_recording = False
        self.is_processing = False
        self.is_closing = False

        self.auto_paste = False
        self.last_hotkey_time = 0.0

        # Запись звука.
        self.stream: Optional[
            sd.InputStream
        ] = None

        self.audio_chunks: list[
            np.ndarray
        ] = []

        self.audio_lock = threading.Lock()

        # Модель.
        self.model: Optional[
            WhisperModel
        ] = None

        # Горячая клавиша.
        self.hotkey_listener: Optional[
            keyboard.GlobalHotKeys
        ] = None

        # Очередь между потоками.
        self.event_queue: queue.Queue[
            tuple[str, object]
        ] = queue.Queue()

        self.create_interface()

        self.root.update_idletasks()

        self.windows = WindowsController(
            self.root
        )

        self.start_hotkey_listener()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_app,
        )

        self.root.after(
            50,
            self.process_events,
        )

        threading.Thread(
            target=self.load_model,
            daemon=True,
        ).start()

    # ========================================================
    # ИНТЕРФЕЙС
    # ========================================================

    def create_interface(self) -> None:
        """Создаёт интерфейс."""

        tk.Label(
            self.root,
            text="VOICE TYPER",
            font=("Segoe UI", 18, "bold"),
            bg="#15171c",
            fg="white",
        ).pack(
            pady=(25, 5)
        )

        tk.Label(
            self.root,
            text="Голосовой ввод текста",
            font=("Segoe UI", 10),
            bg="#15171c",
            fg="#777d89",
        ).pack(
            pady=(0, 12)
        )

        self.status_label = tk.Label(
            self.root,
            text="Загрузка модели...",
            font=("Segoe UI", 11),
            bg="#15171c",
            fg="#f0c75e",
        )
        self.status_label.pack(
            pady=(0, 18)
        )

        self.record_button = tk.Button(
            self.root,
            text="Начать запись",
            command=(
                self.toggle_recording_from_button
            ),
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
            width=20,
            height=2,
        )
        self.record_button.pack()

        tk.Label(
            self.root,
            text="F8 — начать или остановить запись",
            font=("Segoe UI", 9),
            bg="#15171c",
            fg="#777d89",
        ).pack(
            pady=(10, 20)
        )

        tk.Label(
            self.root,
            text="Распознанный текст:",
            font=("Segoe UI", 10),
            bg="#15171c",
            fg="#a5a9b3",
        ).pack(
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
            fg="white",
            insertbackground="white",
            selectbackground="#3478f6",
            relief="flat",
            borderwidth=0,
            wrap="word",
            padx=12,
            pady=12,
        )
        self.result_text.pack(
            padx=30
        )

    # ========================================================
    # СОБЫТИЯ
    # ========================================================

    def process_events(self) -> None:
        """Обрабатывает очередь событий."""

        if self.is_closing:
            return

        # Теперь этот вызов безопасен,
        # даже если Windows вернёт NULL.
        self.windows.update_last_external_window()

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
                    str(event_data)
                )

            elif event_name == "hotkey":
                self.toggle_recording_from_hotkey(
                    int(event_data or 0)
                )

            elif event_name == "transcription_finished":
                self.show_result(
                    str(event_data)
                )

            elif event_name == "transcription_error":
                self.show_transcription_error(
                    str(event_data)
                )

        self.root.after(
            50,
            self.process_events,
        )

    # ========================================================
    # МОДЕЛЬ
    # ========================================================

    def load_model(self) -> None:
        """Загружает Whisper."""

        try:
            print("Загрузка модели Whisper...")

            self.model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )

            print("Модель Whisper загружена")

            self.event_queue.put(
                ("model_loaded", None)
            )

        except Exception as error:
            self.event_queue.put(
                ("model_error", str(error))
            )

    def model_loaded(self) -> None:
        """Активирует запись."""

        self.status_label.configure(
            text="Готов к записи",
            fg="#72d98b",
        )

        self.record_button.configure(
            state=tk.NORMAL
        )

    def show_model_error(
        self,
        error_text: str,
    ) -> None:
        """Показывает ошибку модели."""

        self.status_label.configure(
            text="Ошибка загрузки модели",
            fg="#ff7777",
        )

        messagebox.showerror(
            "Ошибка модели",
            (
                "Не удалось загрузить модель.\n\n"
                f"{error_text}"
            ),
        )

    # ========================================================
    # ГОРЯЧАЯ КЛАВИША
    # ========================================================

    def start_hotkey_listener(self) -> None:
        """Запускает F8."""

        self.hotkey_listener = (
            keyboard.GlobalHotKeys(
                {
                    HOTKEY: self.hotkey_callback,
                }
            )
        )

        self.hotkey_listener.start()

    def hotkey_callback(self) -> None:
        """Вызывается при нажатии F8."""

        current_time = time.monotonic()

        if (
            current_time
            - self.last_hotkey_time
            < 0.4
        ):
            return

        self.last_hotkey_time = current_time

        raw_window = (
            ctypes.windll.user32
            .GetForegroundWindow()
        )

        if raw_window:
            foreground_window = int(
                raw_window
            )
        else:
            foreground_window = 0

        self.event_queue.put(
            (
                "hotkey",
                foreground_window,
            )
        )

    # ========================================================
    # МИКРОФОН
    # ========================================================

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """Получает звук с микрофона."""

        if status:
            print(
                "Статус микрофона:",
                status,
            )

        with self.audio_lock:
            self.audio_chunks.append(
                indata.copy()
            )

    def start_recording(self) -> None:
        """Начинает запись."""

        if self.model is None:
            return

        if self.is_processing:
            return

        try:
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
        """Останавливает запись."""

        if self.stream is None:
            return

        try:
            self.stream.stop()
            self.stream.close()

        except Exception as error:
            print(
                "Ошибка остановки микрофона:",
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
            if self.audio_chunks:
                audio = np.concatenate(
                    self.audio_chunks,
                    axis=0,
                )
            else:
                audio = np.empty(
                    0,
                    dtype=np.float32,
                )

            self.audio_chunks.clear()

        audio = audio.reshape(-1).astype(
            np.float32,
            copy=False,
        )

        minimum_samples = int(
            SAMPLE_RATE
            * MIN_RECORDING_SECONDS
        )

        if audio.size < minimum_samples:
            self.status_label.configure(
                text="Запись слишком короткая",
                fg="#ff7777",
            )

            self.record_button.configure(
                state=tk.NORMAL
            )
            return

        self.is_processing = True

        self.status_label.configure(
            text="Распознавание...",
            fg="#f0c75e",
        )

        threading.Thread(
            target=self.transcribe_audio,
            args=(audio,),
            daemon=True,
        ).start()

    # ========================================================
    # РАСПОЗНАВАНИЕ
    # ========================================================

    def transcribe_audio(
        self,
        audio: np.ndarray,
    ) -> None:
        """Преобразует звук в текст."""

        try:
            if self.model is None:
                raise RuntimeError(
                    "Модель не загружена"
                )

            segments, _ = self.model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            text_parts: list[str] = []

            for segment in segments:
                clean_text = (
                    segment.text.strip()
                )

                if clean_text:
                    text_parts.append(
                        clean_text
                    )

            result = " ".join(
                text_parts
            )

            result = " ".join(
                result.split()
            )

            print(
                "Результат:",
                result,
            )

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
        """Показывает распознанный текст."""

        self.is_processing = False

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
                state=tk.NORMAL
            )

            self.auto_paste = False
            return

        self.result_text.insert(
            tk.END,
            result,
        )

        # Всегда сохраняем в буфер обмена.
        pyperclip.copy(result)

        self.record_button.configure(
            state=tk.NORMAL
        )

        if self.auto_paste:
            self.status_label.configure(
                text="Возвращаю целевое окно...",
                fg="#f0c75e",
            )

            self.root.after(
                150,
                lambda: self.restore_and_paste(
                    result
                ),
            )

        else:
            self.status_label.configure(
                text="Текст скопирован",
                fg="#72d98b",
            )

    def show_transcription_error(
        self,
        error_text: str,
    ) -> None:
        """Показывает ошибку распознавания."""

        self.is_processing = False
        self.auto_paste = False

        self.record_button.configure(
            state=tk.NORMAL
        )

        self.status_label.configure(
            text="Ошибка распознавания",
            fg="#ff7777",
        )

        messagebox.showerror(
            "Ошибка распознавания",
            (
                "Не удалось распознать речь.\n\n"
                f"{error_text}"
            ),
        )

    # ========================================================
    # ВСТАВКА
    # ========================================================

    def restore_and_paste(
        self,
        result: str,
    ) -> None:
        """Возвращает целевое окно."""

        restored = (
            self.windows.restore_target_window()
        )

        if not restored:
            self.status_label.configure(
                text=(
                    "Не удалось вернуть окно; "
                    "текст находится в буфере"
                ),
                fg="#f0c75e",
            )

            self.auto_paste = False
            return

        self.root.after(
            250,
            lambda: self.paste_into_target(
                result
            ),
        )

    def paste_into_target(
        self,
        result: str,
    ) -> None:
        """Вставляет текст в целевое окно."""

        if not self.windows.is_target_foreground():
            restored = (
                self.windows.restore_target_window()
            )

            if not restored:
                self.status_label.configure(
                    text=(
                        "Фокус потерян; "
                        "текст находится в буфере"
                    ),
                    fg="#f0c75e",
                )

                self.auto_paste = False
                return

        paste_command_sent = (
            self.windows.send_paste_shortcut()
        )

        if paste_command_sent:
            self.status_label.configure(
                text="Команда вставки отправлена",
                fg="#72d98b",
            )

        else:
            typed_directly = (
                self.windows.send_unicode_text(
                    result
                )
            )

            if typed_directly:
                self.status_label.configure(
                    text="Текст введён напрямую",
                    fg="#72d98b",
                )
            else:
                self.status_label.configure(
                    text=(
                        "Автоввод не сработал; "
                        "текст находится в буфере"
                    ),
                    fg="#f0c75e",
                )

        self.auto_paste = False

    # ========================================================
    # УПРАВЛЕНИЕ
    # ========================================================

    def toggle_recording_from_button(
        self,
    ) -> None:
        """Управляет записью через кнопку."""

        self.auto_paste = False
        self.toggle_recording()

    def toggle_recording_from_hotkey(
        self,
        foreground_window: int,
    ) -> None:
        """Управляет записью через F8."""

        if self.is_processing:
            return

        # Целевое окно выбираем только
        # при начале новой записи.
        if not self.is_recording:
            self.auto_paste = (
                self.windows.select_target_window(
                    foreground_window
                )
            )

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
        """Закрывает приложение."""

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
        """Запускает интерфейс."""

        self.root.mainloop()


if __name__ == "__main__":
    app = VoiceTyperApp()
    app.run()