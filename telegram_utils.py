import telegram

TELEGRAM_SETTINGS = {
    "TOKEN": "7990774098:AAGBVNNdJdIgIBLA22-4b6ojHnYhNGW7OO8",
    "CHAT_ID": "1710874541"
}

def get_bot_instance():
    return telegram.Bot(token=TELEGRAM_SETTINGS["TOKEN"])

async def send_telegram_message(text: str):
    bot = get_bot_instance()
    try:
        # Приводим chat_id и текст к нужным типам
        text = str(text)
        # Принудительно кодируем и декодируем текст в UTF-8
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        await bot.send_message(chat_id=int(TELEGRAM_SETTINGS["CHAT_ID"]), text=text)
        print("Сообщение отправлено в Telegram")
    except Exception as e:
        print("Ошибка при отправке сообщения:", e)

async def send_telegram_photo(photo_path: str, caption: str = None):
    bot = get_bot_instance()
    try:
        with open(photo_path, 'rb') as photo:
            if caption is not None:
                caption = str(caption)
                caption = caption.encode('utf-8', errors='replace').decode('utf-8')
            await bot.send_photo(chat_id=int(TELEGRAM_SETTINGS["CHAT_ID"]),
                                 photo=photo,
                                 caption=caption)
        print("Изображение отправлено в Telegram")
    except Exception as e:
        print("Ошибка при отправке изображения:", e)
