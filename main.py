import tkinter as tk


class VoiceTyperApp:
    def __init__(self) -> None:
        # Создаём главное окно
        self.root = tk.Tk()

        # Настройки окна
        self.root.title("Voice Typer")
        self.root.geometry("400x260")
        self.root.resizable(False, False)
        self.root.configure(bg="#15171c")

        # Переменная, которая показывает, идёт ли запись
        self.is_recording = False

        # Заголовок
        self.title_label = tk.Label(
            self.root,
            text="VOICE TYPER",
            font=("Segoe UI", 18, "bold"),
            bg="#15171c",
            fg="white",
        )
        self.title_label.pack(pady=(30, 8))

        # Статус приложения
        self.status_label = tk.Label(
            self.root,
            text="Готов к записи",
            font=("Segoe UI", 11),
            bg="#15171c",
            fg="#a5a9b3",
        )
        self.status_label.pack(pady=(0, 25))

        # Кнопка записи
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

    def toggle_recording(self) -> None:
        """Переключает состояние записи."""

        self.is_recording = not self.is_recording

        if self.is_recording:
            self.record_button.configure(
                text="Остановить запись",
                bg="#d94a4a",
                activebackground="#b83b3b",
            )
            self.status_label.configure(
                text="Идёт запись...",
                fg="#ff7777",
            )
        else:
            self.record_button.configure(
                text="Начать запись",
                bg="#3478f6",
                activebackground="#2864d7",
            )
            self.status_label.configure(
                text="Запись остановлена",
                fg="#a5a9b3",
            )

    def run(self) -> None:
        """Запускает главный цикл приложения."""

        self.root.mainloop()


if __name__ == "__main__":
    app = VoiceTyperApp()
    app.run()