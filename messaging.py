import os
import asyncio
from telegram_utils import send_telegram_message, send_telegram_photo

def send_message(text: str):
    """
    Отправляет сообщение через Telegram.
    """
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(send_telegram_message(text))
    except Exception as e:
        print(f"[ERROR] Ошибка отправки сообщения: {e}")
    finally:
        loop.close()

def send_photo(photo_path: str, caption: str = None):
    """
    Отправляет фотографию через Telegram.
    Проверяет существование файла перед отправкой.
    """
    if not os.path.exists(photo_path):
        print(f"[ERROR] Файл {photo_path} не найден!")
        return
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(send_telegram_photo(photo_path, caption))
    except Exception as e:
        print(f"[ERROR] Ошибка отправки фото: {e}")
    finally:
        loop.close()
