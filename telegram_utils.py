import os
import telegram
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TELEGRAM_SETTINGS = {
    "TOKEN": os.getenv("TELEGRAM_TOKEN"),
    "CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
}

if not TELEGRAM_SETTINGS["TOKEN"] or not TELEGRAM_SETTINGS["CHAT_ID"]:
    raise ValueError(
        "Environment variables TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set"
    )

def get_bot_instance() -> telegram.Bot:
    """
    Возвращает экземпляр Telegram-бота.
    """
    return telegram.Bot(token=TELEGRAM_SETTINGS["TOKEN"])

async def send_telegram_message(text: str) -> None:
    """
    Отправляет сообщение в Telegram.
    """
    bot = get_bot_instance()
    try:
        text = str(text)
        # Принудительное преобразование текста в UTF-8
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        await bot.send_message(chat_id=int(TELEGRAM_SETTINGS["CHAT_ID"]), text=text)
        logger.info("Сообщение отправлено в Telegram: %s", text)
    except Exception as e:
        logger.error("Ошибка при отправке сообщения: %s", e)

async def send_telegram_photo(photo_path: str, caption: Optional[str] = None) -> None:
    """
    Отправляет фотографию в Telegram с необязательной подписью.
    """
    bot = get_bot_instance()
    try:
        with open(photo_path, 'rb') as photo:
            if caption is not None:
                caption = str(caption)
                caption = caption.encode('utf-8', errors='replace').decode('utf-8')
            await bot.send_photo(chat_id=int(TELEGRAM_SETTINGS["CHAT_ID"]),
                                 photo=photo,
                                 caption=caption)
        logger.info("Изображение отправлено в Telegram: %s", photo_path)
    except Exception as e:
        logger.error("Ошибка при отправке изображения: %s", e)
