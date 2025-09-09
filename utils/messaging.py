import os
import asyncio
import logging
from utils.telegram_utils import send_telegram_message, send_telegram_photo

logger = logging.getLogger(__name__)

def send_message(text: str) -> None:
    """
    Отправляет сообщение через Telegram.
    """
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(send_telegram_message(text))
        logger.info("Сообщение успешно отправлено: %s", text)
    except Exception as e:
        logger.error("Ошибка отправки сообщения: %s", e)
    finally:
        loop.close()

def send_photo(photo_path: str, caption: str = None) -> None:
    """
    Отправляет фотографию через Telegram. Проверяет существование файла перед отправкой.
    """
    if not os.path.exists(photo_path):
        logger.error("Файл %s не найден!", photo_path)
        return
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(send_telegram_photo(photo_path, caption))
        logger.info("Фото успешно отправлено: %s", photo_path)
    except Exception as e:
        logger.error("Ошибка отправки фото: %s", e)
    finally:
        loop.close()
