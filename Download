#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import logging
import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

import aiohttp
import yt_dlp
from yt_dlp.utils import DownloadError

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
)

# ================== Настройки логирования (выключаем лишние логи) ==================
# Уровень по умолчанию: WARNING — чтобы не засорять логами.
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.WARNING
)
logger = logging.getLogger(__name__)
# Уменьшим логирование библиотек, которые шумят
for noisy in ('httpx', 'telegram', 'yt_dlp', 'urllib3', 'aiohttp'):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ================== Конфигурация ==================
# Токен: рекомендуется задавать через переменные окружения.
BOT_TOKEN = os.getenv("BOT_TOKEN", "8193216642:AAEqSR5MrWT9Y-aRqnt49fA34x7TQsWfETI")
# Админ (без @)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "nonamekakbi")

# Путь к проекту / загрузкам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, 'downloads')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Место ffmpeg (у тебя внутри .venv распаковано)
FFMPEG_DIR = os.path.join(BASE_DIR, 'ffmpeg-2025-09-04-git-2611874a50-full_build', 'bin')
if os.name == 'nt':
    FFMPEG_BIN = os.path.join(FFMPEG_DIR, 'ffmpeg.exe')
else:
    FFMPEG_BIN = os.path.join(FFMPEG_DIR, 'ffmpeg')

if not (os.path.exists(FFMPEG_BIN) or os.path.isdir(FFMPEG_DIR)):
    logger.warning(f"ffmpeg не найден по предполагаемому пути: {FFMPEG_BIN}. "
                   "Укажите путь в FFMPEG_DIR, если нужно.")

# Лимит Telegram для загрузки файлов (оставляем 50MB)
TELEGRAM_FILE_LIMIT = 50 * 1024 * 1024  # байты

# Глобальная aiohttp сессия (создаётся лениво)
AIOHTTP_SESSION: Optional[aiohttp.ClientSession] = None


# ================== Утилиты ==================
def safe_filename(name: str) -> str:
    import re
    cleaned = re.sub(r'[\\/:*?"<>|\n\r]+', '_', name).strip()
    return cleaned[:200]


def is_url(text: str) -> Optional[str]:
    if not text:
        return None
    pattern = re.compile(
        r'(https?://[^\s]+|www\.[^\s]+|vm\.tiktok\.com/[^\s]+|youtu\.be/[^\s]+)',
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return None
    url = m.group(0)
    if url.startswith('www.'):
        url = 'http://' + url
    if url.startswith('vm.tiktok.com') or url.startswith('youtu.be'):
        if not url.startswith('http'):
            url = 'https://' + url
    return url


def resolve_url_sync(url: str, timeout: int = 10) -> str:
    """
    Синхронно разрешаем короткие ссылки (используется в threadpool).
    """
    try:
        import requests
        resp = requests.get(url, allow_redirects=True, timeout=timeout)
        return resp.url
    except Exception as e:
        logger.debug(f"resolve_url failed for {url}: {e}")
        return url


async def get_aiohttp_session() -> aiohttp.ClientSession:
    global AIOHTTP_SESSION
    if AIOHTTP_SESSION is None or AIOHTTP_SESSION.closed:
        AIOHTTP_SESSION = aiohttp.ClientSession()
    return AIOHTTP_SESSION


def split_long_text(text: str, limit: int = 4000) -> List[str]:
    """
    Разбивает длинный текст на части <= limit (по переносам, иначе по символам).
    """
    if len(text) <= limit:
        return [text]
    parts = []
    cur = ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > limit:
            if cur:
                parts.append(cur)
            cur = line
            if len(cur) > limit:
                # очень длинная строка — режем по частям
                for i in range(0, len(cur), limit):
                    parts.append(cur[i:i+limit])
                cur = ""
        else:
            cur += line
    if cur:
        parts.append(cur)
    return parts


# ================== Сессия логов (в памяти) ==================
# Ключ: username_lower (если есть username), иначе "id_<user_id>"
SESSION_LOGS: Dict[str, List[Dict[str, Any]]] = {}
SESSION_LOGS_LOCK = asyncio.Lock()


async def append_session_log(user, incoming_text: str, bot_text: str, extra: Optional[Dict] = None):
    """
    Сохраняет запись во временный лог в памяти.
    """
    try:
        if user:
            uname = user.username or f"id_{user.id}"
            key = uname.lower()
        else:
            key = "unknown"
        entry = {
            "time": datetime.utcnow().isoformat(timespec='seconds'),
            "user_id": getattr(user, "id", None),
            "username": getattr(user, "username", None),
            "incoming": incoming_text,
            "response": bot_text,
            "extra": extra or {}
        }
        async with SESSION_LOGS_LOCK:
            SESSION_LOGS.setdefault(key, []).append(entry)
    except Exception as e:
        logger.debug(f"append_session_log failed: {e}")


# ================== Класс VideoDownloader (обёртка yt_dlp) ==================
class VideoDownloader:
    def __init__(self, download_dir: str, ffmpeg_location: Optional[str] = None):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.base_outtmpl = os.path.join(self.download_dir, '%(id)s.%(ext)s')
        self.ffmpeg_location = ffmpeg_location or FFMPEG_DIR

        self.ydl_common = {
            'outtmpl': self.base_outtmpl,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }

    def _get_common_opts(self):
        opts = self.ydl_common.copy()
        opts['ffmpeg_location'] = self.ffmpeg_location
        opts['merge_output_format'] = 'mp4'
        return opts

    def get_video_info(self, url: str):
        """
        Возвращает info dict (yt_dlp.extract_info(..., download=False))
        Выполняется синхронно (предназначено для запуска в threadpool).
        """
        try:
            resolved = resolve_url_sync(url)
            ydl_opts = self._get_common_opts()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(resolved, download=False)
                return info
        except Exception as e:
            logger.exception(f"Ошибка получения информации о видео: {e}")
            return None

    def youtube_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Использует ytsearch для поиска на YouTube. Возвращает упрощённый список словарей.
        Выполняется в threadpool.
        """
        try:
            ydl_opts = self._get_common_opts()
            ydl_opts.update({'quiet': True})
            search = f"ytsearch{limit}:{query}"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                results = ydl.extract_info(search, download=False)
                entries = results.get('entries') if isinstance(results, dict) else None
                if not entries:
                    return []
                out = []
                for e in entries:
                    if not e:
                        continue
                    out.append({
                        'id': e.get('id'),
                        'title': e.get('title'),
                        'uploader': e.get('uploader'),
                        'duration': e.get('duration'),
                        'webpage_url': e.get('webpage_url') or e.get('url'),
                        'thumbnail': e.get('thumbnail'),
                    })
                return out
        except Exception as e:
            logger.exception(f"youtube_search failed: {e}")
            return []

    def _find_downloaded_file(self, info: dict, prefer_exts=None) -> Optional[str]:
        try:
            for key in ('requested_downloads', '_filename', 'filepath', 'filename'):
                v = info.get(key)
                if isinstance(v, list):
                    for r in v:
                        fp = r.get('filepath') or r.get('filename')
                        if fp and os.path.exists(fp):
                            return fp
                elif isinstance(v, str):
                    if os.path.exists(v):
                        return v

            vid = info.get('id')
            if vid:
                if prefer_exts:
                    for ext in prefer_exts:
                        candidate = os.path.join(self.download_dir, f"{vid}.{ext}")
                        if os.path.exists(candidate):
                            return candidate
                globp = os.path.join(self.download_dir, f"{vid}.*")
                matches = sorted(glob.glob(globp), key=os.path.getmtime, reverse=True)
                if matches:
                    return matches[0]

            files = sorted(glob.glob(os.path.join(self.download_dir, '*')), key=os.path.getmtime, reverse=True)
            if files:
                return files[0]
        except Exception as e:
            logger.debug(f"_find_downloaded_file error: {e}")
        return None

    def _attempt_download(self, resolved_url: str, ydl_opts: dict):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(resolved_url, download=True)
                return info, None
        except DownloadError as e:
            logger.warning(f"yt_dlp DownloadError for {resolved_url} with format={ydl_opts.get('format')}: {e}")
            return None, e
        except Exception as e:
            logger.exception(f"Unexpected error in _attempt_download: {e}")
            return None, e

    def download_video(self, url: str, format_selector: Optional[str] = None, as_audio: bool = False) -> Optional[str]:
        """
        Скачивает (запускается в threadpool): если as_audio=True — конвертирует в mp3.
        Возвращает путь к файлу или None.
        """
        try:
            resolved = resolve_url_sync(url)
            ydl_opts = self._get_common_opts()

            if format_selector:
                ydl_opts['format'] = format_selector
            else:
                ydl_opts['format'] = 'best'

            if as_audio:
                ydl_opts['format'] = 'bestaudio/best'
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }]
                ydl_opts['outtmpl'] = os.path.join(self.download_dir, '%(id)s.%(ext)s')

            info, exc = self._attempt_download(resolved, ydl_opts)
            if info and not exc:
                prefer_exts = ['mp3'] if as_audio else ['mp4', 'mkv', 'webm', 'mov', 'm4v']
                fp = self._find_downloaded_file(info, prefer_exts=prefer_exts)
                if fp:
                    return fp
                try:
                    candidate = info.get('_filename') or (info.get('requested_downloads') or [{}])[0].get('filepath')
                    if candidate and os.path.exists(candidate):
                        return candidate
                except Exception:
                    pass
                files = sorted(glob.glob(os.path.join(self.download_dir, '*')), key=os.path.getmtime, reverse=True)
                if files:
                    return files[0]

            logger.error("Не удалось скачать видео/аудио: все форматы завершились неудачей")
            return None
        except Exception as e:
            logger.exception(f"Ошибка скачивания: {e}")
            return None

    def format_file_size(self, size_bytes):
        if not size_bytes:
            return "N/A"
        try:
            size_bytes = int(size_bytes)
        except:
            return "N/A"
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def move_to_trash(self, file_path):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Не удалось удалить файл {file_path}: {e}")
            return False


# Инициализация
video_downloader = VideoDownloader(DOWNLOAD_DIR, ffmpeg_location=FFMPEG_DIR)


# ================== Хэндлеры ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = ("Привет! Я могу:\n"
           "• Скачать видео (YouTube/TikTok и др.) — отправь ссылку и выбери качество.\n"
           "• Скачать аудио (MP3) из видео.\n"
           "• Поиск музыки: отправь 'найти <текст>'.\n"
           "• Конвертация валют: /convert 100 USD EUR\n"
           "• Админ: Узнать активность @username (только для админа).\n\n"
           "Команда /help — список команд.")
    sent = await update.message.reply_text(txt)
    await append_session_log(update.effective_user, "/start", txt)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = ("Команды и примеры:\n"
           "/start — стартовое сообщение\n"
           "/help — помощь\n"
           "/convert <сумма> <из> <в> — конвертация валют (пример: /convert 100 USD EUR)\n"
           "найти <текст> — поиск треков через YouTube (пример: найти Bad Guy)\n"
           "Отправь ссылку на видео — появится выбор качества / MP3\n"
           "Узнать активность @username — (только админ) выводит временный лог активности.")
    sent = await update.message.reply_text(txt)
    await append_session_log(update.effective_user, "/help", txt)


async def show_formats_keyboard_for_video(info: dict, message_id: int):
    """
    Возвращает (text, InlineKeyboardMarkup)
    """
    title = info.get('title') or info.get('uploader') or 'Unknown'
    duration = info.get('duration')
    durtext = f"{int(duration)//60}:{int(duration)%60:02d}" if duration else "N/A"
    text = f"🎬 {title}\n⏱ {durtext}\nВыберите качество (или MP3):"

    heights = set()
    formats = info.get('formats', []) or []
    max_h = 0
    for f in formats:
        h = f.get('height')
        try:
            if h:
                heights.add(int(h))
                if int(h) > max_h:
                    max_h = int(h)
        except Exception:
            continue

    standard = [144, 240, 360, 480, 720, 1080, 1440, 2160]
    avail = [h for h in standard if h <= (max_h or 2160)]
    if not avail:
        avail = [144, 360, 720]

    keyboard = []
    keyboard.append([InlineKeyboardButton("🎵 MP3 (аудио)", callback_data=f"dl_mp3|{message_id}")])
    row = []
    for idx, h in enumerate(avail, 1):
        row.append(InlineKeyboardButton(f"{h}p", callback_data=f"dl_sel|{message_id}|{h}"))
        if idx % 3 == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    return text, InlineKeyboardMarkup(keyboard)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Основной обработчик текстовых сообщений (ссылки, 'найти', 'конверт', 'Узнать активность ...' и т.д.)
    """
    if not update.message or not update.message.text:
        return

    text_raw = update.message.text.strip()
    text_lower = text_raw.lower()

    # 1) Если запрос на получение активности: "Узнать активность @username"
    m_act = re.match(r'^\s*узнать\s+активн(?:ость|ость)\s+@?([A-Za-z0-9_]{1,32})\s*$', text_lower, re.IGNORECASE)
    if m_act:
        # обработаем через отдельную функцию
        await handle_activity_request_message(update, context, target_username=m_act.group(1))
        return

    # 2) Пытаемся найти URL
    url = None
    if update.message.entities:
        for ent in update.message.entities:
            try:
                if ent.type == 'url':
                    url = update.message.text[ent.offset: ent.offset + ent.length]
                    break
                if ent.type == 'text_link':
                    url = ent.url
                    break
            except Exception:
                continue
    if not url:
        url = is_url(text_raw)

    if url:
        # Получаем информацию о видео в threadpool
        typing_msg = await update.message.reply_text("🔎 Получаю информацию о видео...")
        await append_session_log(update.effective_user, text_raw, "🔎 Получаю информацию о видео...")
        info = await asyncio.to_thread(video_downloader.get_video_info, url)
        if not info:
            txt = "❌ Не удалось получить информацию о видео."
            await update.message.reply_text(txt)
            await append_session_log(update.effective_user, text_raw, txt)
            return

        msg_id = update.message.message_id
        context.user_data[f'video_url_{msg_id}'] = url
        context.user_data[f'video_info_{msg_id}'] = info

        text, markup = await show_formats_keyboard_for_video(info, msg_id)
        await update.message.reply_text(text, reply_markup=markup)
        await append_session_log(update.effective_user, text_raw, text)
        return

    # 3) Поиск музыки: "найти <query>"
    if text_lower.startswith('найти'):
        query = re.sub(r'^(найти)\s*', '', text_raw, flags=re.IGNORECASE).strip()
        if not query:
            txt = "Укажите поисковую строку после 'найти'."
            await update.message.reply_text(txt)
            await append_session_log(update.effective_user, text_raw, txt)
            return

        loading = await update.message.reply_text("🔍 Ищу в YouTube...")
        await append_session_log(update.effective_user, text_raw, "🔍 Ищу в YouTube...")
        results = await asyncio.to_thread(video_downloader.youtube_search, query, 6)
        if not results:
            txt = "❌ Результатов не найдено."
            await update.message.reply_text(txt)
            await append_session_log(update.effective_user, text_raw, txt)
            return

        key = f"yt_search_{update.message.message_id}"
        context.user_data[key] = results

        text = "🎵 Результаты поиска:\n\n"
        keyboard = []
        for i, r in enumerate(results, 1):
            title = r.get('title') or 'Unknown'
            uploader = r.get('uploader') or ''
            dur = r.get('duration')
            durstr = f" [{int(dur)//60}:{int(dur)%60:02d}]" if dur else ""
            text += f"{i}. {title} — {uploader}{durstr}\n"
            keyboard.append([InlineKeyboardButton(f"{title}", callback_data=f"sendtrack|{update.message.message_id}|{i-1}")])

        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        await append_session_log(update.effective_user, text_raw, text)
        return

    # 4) Конвертация валют: "/convert 100 USD EUR" или "конверт 100 USD EUR"
    m_conv = re.match(r'^\s*конверт(?:ация)?\s+([0-9\.,]+)\s+([A-Za-z]{3})\s+([A-Za-z]{3})\s*$', text_raw, re.IGNORECASE)
    if not m_conv:
        # также принимаем английский /convert в виде текста
        m_conv = re.match(r'^\s*/?convert\s+([0-9\.,]+)\s+([A-Za-z]{3})\s+([A-Za-z]{3})\s*$', text_raw, re.IGNORECASE)
    if m_conv:
        amount = m_conv.group(1).replace(',', '.')
        frm = m_conv.group(2).upper()
        to = m_conv.group(3).upper()
        try:
            amt = float(amount)
        except:
            txt = "Не удалось распознать сумму. Пример: /convert 100 USD EUR"
            await update.message.reply_text(txt)
            await append_session_log(update.effective_user, text_raw, txt)
            return

        await update.message.reply_text(f"🔁 Конвертирую {amt} {frm} → {to} ...")
        await append_session_log(update.effective_user, text_raw, f"🔁 Конвертирую {amt} {frm} → {to} ...")
        result = await convert_currency(amt, frm, to)
        if not result:
            txt = "❌ Не удалось получить курс (проверьте код валюты)."
            await update.message.reply_text(txt)
            await append_session_log(update.effective_user, text_raw, txt)
            return
        converted, rate = result
        txt = f"{amt} {frm} = {converted:.4f} {to}\nКурс: 1 {frm} = {rate:.6f} {to}"
        await update.message.reply_text(txt)
        await append_session_log(update.effective_user, text_raw, txt)
        return

    # 5) Ничего не понял — показываем обычное меню команд
    txt = ("Не понял запрос. Доступные команды:\n"
           "/start — старт\n"
           "/help — помощь\n"
           "найти <текст> — поиск треков\n"
           "Отправь ссылку — я покажу качества для видео\n"
           "/convert <сумма> <из> <в> — конвертация валют\n"
           "Узнать активность @username — (только админ)\n")
    await update.message.reply_text(txt)
    await append_session_log(update.effective_user, text_raw, txt)


async def handle_activity_request_message(update: Update, context: ContextTypes.DEFAULT_TYPE, target_username: str):
    """
    Обработчик запроса активности, если вызван в чате как обычное сообщение.
    """
    caller = update.effective_user
    if not caller or (caller.username or "").lower() != ADMIN_USERNAME.lower():
        txt = "⚠️ Команда доступна только администратору."
        await update.message.reply_text(txt)
        await append_session_log(caller, f"Узнать активность @{target_username}", txt)
        return

    await send_activity_for_username(update, context, target_username)


async def send_activity_for_username(update_or_query, context: ContextTypes.DEFAULT_TYPE, target_username: str):
    """
    Общая функция для отправки активности по username (без @).
    После отправки — удаляем лог для этого пользователя (чтобы не хранить).
    """
    target = (target_username or "").lstrip('@').lower()
    async with SESSION_LOGS_LOCK:
        entries = SESSION_LOGS.get(target)
        if not entries:
            txt = f"ℹ️ Пользователь @{target_username} не использовал бота или лог пуст."
            # если update_or_query — это Update, отправляем в update_or_query.message
            if isinstance(update_or_query, Update) and update_or_query.message:
                await update_or_query.message.reply_text(txt)
            else:
                # возможно это callback query
                try:
                    await update_or_query.message.reply_text(txt)
                except Exception:
                    pass
            return

        # Формируем текст
        lines = []
        for e in entries:
            ts = e.get('time')
            uname = e.get('username') or f"id_{e.get('user_id')}"
            link = f"https://t.me/{uname}" if e.get('username') else ""
            incoming = e.get('incoming') or ""
            response = e.get('response') or ""
            lines.append(f"[{ts}] <a href='{link}'>@{uname}</a>\nUser → {incoming}\nBot  → {response}\n")

        big = "\n".join(lines)
        # Отправляем частями (HTML)
        parts = split_long_text(big, limit=4000)
        for p in parts:
            if isinstance(update_or_query, Update) and update_or_query.message:
                await update_or_query.message.reply_text(p, parse_mode='HTML', disable_web_page_preview=True)
            else:
                try:
                    await update_or_query.message.reply_text(p, parse_mode='HTML', disable_web_page_preview=True)
                except Exception:
                    pass

        # очищаем лог для этого пользователя (временное хранение)
        del SESSION_LOGS[target]


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    user = update.effective_user

    # Логируем само действие пользователя (нажатие)
    await append_session_log(user, f"callback_pressed: {data}", "(нажатие кнопки)")

    try:
        if data.startswith('dl_sel|'):
            parts = data.split('|', 2)
            if len(parts) != 3:
                await query.edit_message_text("Неверный запрос.")
                await append_session_log(user, data, "Неверный запрос.")
                return
            _, msgid, height = parts
            url = context.user_data.get(f'video_url_{msgid}')
            info = context.user_data.get(f'video_info_{msgid}')
            if not url or not info:
                txt = "Информация о видео недоступна. Повторите отправку ссылки."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return

            fmt_selector = f"bestvideo[height<={height}]+bestaudio/best"
            est_size = None
            for f in info.get('formats', []):
                try:
                    if f.get('height') and int(f.get('height') or 0) <= int(height):
                        est_size = f.get('filesize') or f.get('filesize_approx')
                        if est_size:
                            break
                except Exception:
                    continue

            est_text = video_downloader.format_file_size(est_size)
            if est_size and est_size > TELEGRAM_FILE_LIMIT:
                txt = f"❌ Примерный размер {est_text} превышает лимит Telegram (50MB). Выберите другое качество."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return

            await query.edit_message_text(f"⏳ Скачиваю {height}p... (ожидаемый размер: {est_text})")
            info_download = await asyncio.to_thread(video_downloader.download_video, url, fmt_selector, False)

            if not info_download or not os.path.exists(info_download):
                txt = "❌ Ошибка скачивания видео в этом качестве."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return

            try:
                size = os.path.getsize(info_download)
                size_text = video_downloader.format_file_size(size)
            except:
                size_text = "N/A"

            if size and size > TELEGRAM_FILE_LIMIT:
                txt = f"❌ Файл получился {size_text}, это больше лимита Telegram (50MB)."
                await query.edit_message_text(txt)
                video_downloader.move_to_trash(info_download)
                await append_session_log(user, data, txt)
                return

            # отправляем видео
            try:
                with open(info_download, 'rb') as f:
                    await query.message.reply_video(video=f)
                txt = f"✅ Видео отправлено ({size_text})."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
            except Exception as e:
                logger.exception(f"Ошибка при отправке видео: {e}")
                txt = "❌ Ошибка при отправке видео."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
            finally:
                video_downloader.move_to_trash(info_download)
            return

        if data.startswith('dl_mp3|'):
            parts = data.split('|', 1)
            if len(parts) != 2:
                await query.edit_message_text("Неверный запрос.")
                await append_session_log(user, data, "Неверный запрос.")
                return
            _, msgid = parts
            url = context.user_data.get(f'video_url_{msgid}')
            info = context.user_data.get(f'video_info_{msgid}')
            if not url or not info:
                txt = "Информация о видео недоступна. Повторите отправку ссылки."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return

            await query.edit_message_text("⏳ Скачиваю аудио (MP3)...")
            filepath = await asyncio.to_thread(video_downloader.download_video, url, None, True)

            if not filepath or not os.path.exists(filepath):
                txt = "❌ Ошибка скачивания/конвертации в MP3. Проверьте, установлен ли ffmpeg в папке проекта."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return

            try:
                size = os.path.getsize(filepath)
                size_text = video_downloader.format_file_size(size)
            except:
                size_text = "N/A"

            if size and size > TELEGRAM_FILE_LIMIT:
                txt = f"❌ Получившийся файл {size_text} превышает лимит Telegram (50MB)."
                await query.edit_message_text(txt)
                video_downloader.move_to_trash(filepath)
                await append_session_log(user, data, txt)
                return

            try:
                with open(filepath, 'rb') as f:
                    await query.message.reply_audio(audio=f)
                txt = f"✅ Аудио отправлено ({size_text})."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
            except Exception as e:
                logger.exception(f"Ошибка при отправке аудио: {e}")
                txt = "❌ Ошибка при отправке аудио."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
            finally:
                video_downloader.move_to_trash(filepath)
            return

        if data.startswith('sendtrack|'):
            parts = data.split('|', 2)
            if len(parts) != 3:
                txt = "Неверный callback для трека."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return
            _, orig_msg_id, idx = parts
            key = f"yt_search_{orig_msg_id}"
            tracks = context.user_data.get(key)
            if not tracks:
                txt = "Результаты поиска устарели. Повторите поиск."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return
            try:
                idx_i = int(idx)
            except:
                txt = "Неверный индекс трека."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return
            if idx_i < 0 or idx_i >= len(tracks):
                txt = "Неверный индекс трека."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return

            track = tracks[idx_i]
            title = track.get('title') or 'Unknown'
            uploader = track.get('uploader') or ''
            video_url = track.get('webpage_url')
            if not video_url:
                txt = "Не удалось получить ссылку на видео."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return

            await query.edit_message_text(f"⬇️ Скачиваю трек: {title} — {uploader} (MP3)...")
            filepath = await asyncio.to_thread(video_downloader.download_video, video_url, None, True)

            if not filepath or not os.path.exists(filepath):
                txt = "❌ Ошибка скачивания трека (MP3)."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
                return

            try:
                size = os.path.getsize(filepath)
                size_text = video_downloader.format_file_size(size)
            except:
                size_text = "N/A"

            if size and size > TELEGRAM_FILE_LIMIT:
                txt = f"❌ Трек {size_text} слишком велик для Telegram (50MB)."
                await query.edit_message_text(txt)
                video_downloader.move_to_trash(filepath)
                await append_session_log(user, data, txt)
                return

            try:
                with open(filepath, 'rb') as f:
                    await query.message.reply_audio(audio=f, title=title, performer=uploader)
                txt = f"✅ Трек отправлен ({size_text})."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
            except Exception as e:
                logger.exception(f"Ошибка при отправке трека: {e}")
                txt = "❌ Ошибка при отправке трека."
                await query.edit_message_text(txt)
                await append_session_log(user, data, txt)
            finally:
                video_downloader.move_to_trash(filepath)
            return

        await query.edit_message_text("Неизвестный callback.")
        await append_session_log(user, data, "Неизвестный callback.")
    except Exception as e:
        logger.exception(f"Ошибка в callback handler: {e}")
        try:
            await query.edit_message_text("⚠️ Произошла внутренняя ошибка при обработке запроса.")
            await append_session_log(user, data, "⚠️ Произошла внутренняя ошибка при обработке запроса.")
        except Exception:
            pass


# ================== Конвертация валют (через exchangerate.host) ==================
async def convert_currency(amount: float, frm: str, to: str) -> Optional[tuple]:
    """
    Выполняет конвертацию через https://api.exchangerate.host/convert
    Возвращает (converted_amount, rate) или None
    """
    try:
        session = await get_aiohttp_session()
        url = "https://api.exchangerate.host/convert"
        params = {"from": frm, "to": to, "amount": amount}
        async with session.get(url, params=params, timeout=10) as resp:
            if resp.status != 200:
                logger.debug(f"convert_currency: status {resp.status}")
                return None
            data = await resp.json()
            if not data or 'result' not in data:
                return None
            result = data.get('result')
            # rate may be in 'info' -> 'rate'
            rate = data.get('info', {}).get('rate') or (result / amount if amount != 0 else None)
            return result, rate
    except Exception as e:
        logger.exception(f"convert_currency failed: {e}")
        return None


# ================== Команды для активности (админ) ==================
async def activity_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработка команды /activity @username
    """
    caller = update.effective_user
    if not caller or (caller.username or "").lower() != ADMIN_USERNAME.lower():
        txt = "⚠️ Команда доступна только администратору."
        await update.message.reply_text(txt)
        await append_session_log(caller, "/activity", txt)
        return

    if not context.args:
        txt = "Укажите username: /activity @username"
        await update.message.reply_text(txt)
        await append_session_log(caller, "/activity", txt)
        return
    target = context.args[0].lstrip('@')
    await send_activity_for_username(update, context, target)


# ================== Глобальная обработка ошибок ==================
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception(f"Unhandled exception: {context.error}")
    try:
        if hasattr(update, 'message') and update.message:
            await update.message.reply_text("⚠️ Произошла ошибка при обработке вашего запроса.")
            await append_session_log(update.effective_user, "(error)", "⚠️ Произошла ошибка при обработке вашего запроса.")
    except Exception:
        pass


# ================== Запуск бота ==================
def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN не задан. Установите переменную окружения BOT_TOKEN и запустите снова.")

    application = Application.builder().token(BOT_TOKEN).build()

    # Команды
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("convert", lambda u, c: asyncio.create_task(help_command(u, c)) if False else None))  # заглушка (в текстовом виде /convert обрабатывается в handle_message)
    application.add_handler(CommandHandler("activity", activity_command))  # /activity @username

    # Callback для кнопок и основной текстовый handler
    application.add_handler(CallbackQueryHandler(handle_callback_query))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.add_error_handler(global_error_handler)

    logger.warning("Бот запущен (логи сведены до WARNING).")
    application.run_polling()


if __name__ == '__main__':
    main()
