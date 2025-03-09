import imaplib
import email
import os
import re
import unicodedata
from email.header import decode_header
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import math

# Конфигурация подключения
USERNAME = 'avtushenko_ue@petroviser.ru'  # Замените на ваш email
PASSWORD = 'gdndtlctruahvfyl'                # Замените на ваш пароль или пароль приложения
IMAP_SERVER = 'imap.yandex.ru'
IMAP_PORT = 993

# Базовая директория для сохранения данных
BASE_DIR = os.path.abspath("yandex_mail_backup")
EMAILS_ROOT = os.path.join(BASE_DIR, "emails")

# Настройка многопоточности: количество потоков
MAX_WORKERS = 10  # задайте нужное количество потоков

# Флаг фильтрации имён файлов (с сохранением русских букв)
FILTER_FILENAMES = True

def sanitize_filename(s, filter_flag=FILTER_FILENAMES):
    """
    Нормализует строку (NFKC), удаляет переводы строки и заменяет все недопустимые символы
    (оставляя буквы, цифры, пробел, точки, тире и нижнее подчёркивание) на '_'.
    """
    if not filter_flag:
        return s
    s = unicodedata.normalize('NFKC', s)
    # Удаляем символы перевода строки и табуляции
    s = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Заменяем все символы, кроме разрешённых, на '_'
    return re.sub(r'[^\w\s\.-]', '_', s).strip()

def ensure_dir(directory):
    """Создаёт директорию, если её ещё нет."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def open_connection():
    """Открывает новое IMAP-соединение и входит в аккаунт."""
    try:
        conn = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        conn.login(USERNAME, PASSWORD)
        conn.select("INBOX")
        return conn
    except Exception as e:
        print("Ошибка открытия соединения:", e)
        return None

def get_email_body(msg):
    """
    Извлекает тело письма в виде текста.
    Предпочтение отдается части с content-type "text/plain", если нет – "text/html".
    """
    body = None
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in content_disposition.lower():
                try:
                    charset = part.get_content_charset() or "utf-8"
                    body = part.get_payload(decode=True).decode(charset, errors="replace")
                    return body
                except Exception:
                    continue
        for part in msg.walk():
            if part.get_content_type() == "text/html" and "attachment" not in str(part.get("Content-Disposition", "")).lower():
                try:
                    charset = part.get_content_charset() or "utf-8"
                    body = part.get_payload(decode=True).decode(charset, errors="replace")
                    return body
                except Exception:
                    continue
    else:
        try:
            charset = msg.get_content_charset() or "utf-8"
            body = msg.get_payload(decode=True).decode(charset, errors="replace")
        except Exception:
            body = msg.get_payload()
    return body if body else ""

def create_email_folder(base_day_dir, subject):
    """
    Создаёт уникальную папку для письма в каталоге base_day_dir.
    Имя папки берётся из темы письма (subject). При совпадении имен добавляется индекс.
    """
    sanitized_subject = sanitize_filename(subject)
    if not sanitized_subject:
        sanitized_subject = "no_subject"
    folder_name = sanitized_subject
    full_path = os.path.join(base_day_dir, folder_name)
    index = 1
    while os.path.exists(full_path):
        folder_name = f"{sanitized_subject}_{index}"
        full_path = os.path.join(base_day_dir, folder_name)
        index += 1
    ensure_dir(full_path)
    return full_path

def process_single_email(conn, email_id):
    """Обрабатывает одно письмо: извлекает данные, формирует структуру папок и сохраняет файлы."""
    try:
        status, msg_data = conn.fetch(email_id, "(RFC822)")
        if status != "OK":
            print(f"Ошибка получения письма с id {email_id.decode('utf-8')}")
            return False

        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        # Определяем дату письма
        date_header = msg.get("Date")
        try:
            dt = parsedate_to_datetime(date_header) if date_header else None
        except Exception as e:
            dt = None

        year = dt.strftime("%Y") if dt else "unknown_year"
        month = dt.strftime("%m") if dt else "unknown_month"
        day = dt.strftime("%d") if dt else "unknown_day"

        # Создаем структуру папок: emails/год/месяц/день
        base_day_dir = os.path.join(EMAILS_ROOT, year, month, day)
        ensure_dir(base_day_dir)

        # Получаем тему письма
        subject = msg.get("Subject", "no_subject")
        decoded_subject, encoding = decode_header(subject)[0]
        if isinstance(decoded_subject, bytes):
            try:
                subject = decoded_subject.decode(encoding if encoding else "utf-8")
            except Exception:
                subject = decoded_subject.decode("utf-8", errors="ignore")
        else:
            subject = str(decoded_subject)

        # Создаем папку для письма
        email_folder = create_email_folder(base_day_dir, subject)

        # Сохраняем тело письма в файл
        body = get_email_body(msg)
        body_filepath = os.path.join(email_folder, "body.txt")
        with open(body_filepath, "w", encoding="utf-8") as f:
            f.write(body)
        # Сохраняем исходное письмо
        eml_filepath = os.path.join(email_folder, "email.eml")
        with open(eml_filepath, "wb") as f:
            f.write(raw_email)
        # Сохраняем вложения
        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = part.get("Content-Disposition", "")
                if "attachment" in content_disposition.lower():
                    filename = part.get_filename()
                    if filename:
                        decoded_filename, enc = decode_header(filename)[0]
                        if isinstance(decoded_filename, bytes):
                            try:
                                decoded_filename = decoded_filename.decode(enc if enc else "utf-8")
                            except Exception:
                                decoded_filename = decoded_filename.decode("utf-8", errors="ignore")
                        sanitized_filename = sanitize_filename(decoded_filename)
                        attachment_path = os.path.join(email_folder, sanitized_filename)
                        base_name, ext = os.path.splitext(sanitized_filename)
                        counter = 1
                        while os.path.exists(attachment_path):
                            attachment_path = os.path.join(email_folder, f"{base_name}_{counter}{ext}")
                            counter += 1
                        with open(attachment_path, "wb") as f:
                            f.write(part.get_payload(decode=True))
        return True
    except Exception as e:
        print(f"Ошибка обработки письма {email_id.decode('utf-8')}: {e}")
        return False

def process_email_subset(email_ids_subset, progress_bar):
    """Функция-воркер: открывает своё соединение и обрабатывает назначенный поднабор email-ID."""
    conn = open_connection()
    if conn is None:
        for _ in email_ids_subset:
            progress_bar.update(1)
        return
    for eid in email_ids_subset:
        process_single_email(conn, eid)
        progress_bar.update(1)
    conn.logout()

def main():
    ensure_dir(BASE_DIR)
    ensure_dir(EMAILS_ROOT)

    # Открываем основное соединение для получения списка email-ID
    main_conn = open_connection()
    if main_conn is None:
        return

    status, data = main_conn.search(None, "ALL")
    if status != "OK":
        print("Ошибка поиска писем.")
        main_conn.logout()
        return

    email_ids = data[0].split()
    total_emails = len(email_ids)
    print(f"Найдено {total_emails} писем.")
    main_conn.logout()

    # Разбиваем список email_ids на подсписки для каждого потока
    chunk_size = math.ceil(total_emails / MAX_WORKERS)
    subsets = [email_ids[i:i + chunk_size] for i in range(0, total_emails, chunk_size)]

    # Создаем общий прогресс-бар
    with tqdm(total=total_emails, desc="Обработка писем", unit="email") as progress_bar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for subset in subsets:
                futures.append(executor.submit(process_email_subset, subset, progress_bar))
            # Ожидаем завершения всех задач
            for future in futures:
                future.result()

if __name__ == "__main__":
    main()
