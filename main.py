"""
Telegram bot: генерация текста (RuT5) + распознавание объектов (YOLOv5)

Автор: адаптировано для RuT5 cointegrated/rut5-base-multitask
"""

import os
import logging
import asyncio
from io import BytesIO
from typing import Optional, Tuple, List, Dict
import time

# ML libs
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import requests

# Telegram
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ---------------------------------------------------------------------------
# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация (берется из окружения)
TELEGRAM_TOKEN = ('7794360700:AAFNjSueLSFobn1se0lkB0OG0NYToFHdZkA')
if not TELEGRAM_TOKEN:
    raise RuntimeError('TELEGRAM_TOKEN не найден в переменных окружения.')

HF_API_TOKEN = os.environ.get('HF_API_TOKEN')
USE_HF_INFERENCE = os.environ.get('USE_HF_INFERENCE', '').lower() in ('1', 'true', 'yes')
MODEL_NAME = os.environ.get('MODEL_NAME', 'cointegrated/rut5-base-multitask')  # RuT5 по умолчанию
YOLO_WEIGHTS = os.environ.get('YOLO_WEIGHTS', '')  # путь к .pt файлу — опционально

# Генеративные параметры по умолчанию
MAX_GEN_TOKENS = int(os.environ.get('MAX_GEN_TOKENS', '120'))
TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.8'))
TOP_P = float(os.environ.get('TOP_P', '0.95'))
TOP_K = int(os.environ.get('TOP_K', '50'))
NUM_RETURN_SEQUENCES = int(os.environ.get('NUM_RETURN_SEQUENCES', '1'))

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'Запуск бота. Устройство для PyTorch: {DEVICE}')

# ---------------------------------------------------------------------------
# Глобальные объекты (ленивая загрузка)
_text_tokenizer: Optional[AutoTokenizer] = None
_text_model: Optional[AutoModelForSeq2SeqLM] = None
_text_pipeline: Optional[pipeline] = None
_yolo_model = None
_model_lock = asyncio.Lock()  # чтобы не загрузить модель дважды при конкурентных запросах

# Простая защита от спама: минимальный интервал между запросами одного пользователя
USER_LAST_TS: Dict[int, float] = {}
MIN_INTERVAL_SECONDS = 1.0
# …весь твой код до глобальных переменных…
USER_CONTEXT: Dict[int, List[str]] = {}
MAX_CONTEXT_WORDS = 100  # лимит памяти

def update_user_context(user_id: int, text: str):
    """Обновление памяти пользователя с лимитом в MAX_CONTEXT_WORDS слов"""
    words = text.split()
    context = USER_CONTEXT.get(user_id, [])
    context += words
    if len(context) > MAX_CONTEXT_WORDS:
        context = context[-MAX_CONTEXT_WORDS:]
    USER_CONTEXT[user_id] = context

def build_prompt(user_id: int, new_message: str) -> str:
    """Создаем prompt для генерации текста с учетом контекста пользователя"""
    context = USER_CONTEXT.get(user_id, [])
    context_text = " ".join(context)
    prompt = f"{context_text}\nПользователь: {new_message}\nБот:"
    return prompt

async def on_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    new_msg = update.message.text.strip()
    now = time.time()
    last = USER_LAST_TS.get(user_id, 0)
    if now - last < MIN_INTERVAL_SECONDS:
        await update.message.reply_text('Не спамьте слишком часто.')
        return
    USER_LAST_TS[user_id] = now

    update_user_context(user_id, new_msg)
    prompt = build_prompt(user_id, new_msg)
    await update.message.chat.send_action('typing')

    try:
        loop = asyncio.get_event_loop()
        if USE_HF_INFERENCE:
            response = await loop.run_in_executor(None, generate_text_hf, prompt)
        else:
            await ensure_text_model_loaded()
            response = await loop.run_in_executor(None, generate_text_local, prompt)

        update_user_context(user_id, response)
        await update.message.reply_text(response)
    except Exception as e:
        logger.exception('Ошибка генерации текста')
        await update.message.reply_text(f'Ошибка: {e}')


# обработка фото осталась, но можно добавить update_user_context с описанием объектов
async def on_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    now = time.time()
    last = USER_LAST_TS.get(user_id, 0)
    if now - last < MIN_INTERVAL_SECONDS:
        await update.message.reply_text('Пожалуйста, не присылайте запросы так часто.')
        return
    USER_LAST_TS[user_id] = now

    photos = update.message.photo
    if not photos:
        await update.message.reply_text('Я не нашел фото в сообщении.')
        return

    photo = photos[-1]
    f = await photo.get_file()
    bio = BytesIO()
    await f.download(out=bio)
    bio.seek(0)
    img_bytes = bio.read()

    await update.message.chat.send_action(action='upload_photo')

    try:
        await ensure_yolo_loaded()
        loop = asyncio.get_event_loop()
        det_list, annotated_bytes = await loop.run_in_executor(None, detect_objects_from_bytes, img_bytes)

        # Обновляем память пользователя описанием фото
        desc = ", ".join([f"{d['class']}({d['confidence']:.2f})" for d in det_list])
        if desc:
            update_user_context(user_id, f"На фото: {desc}")

        if not det_list:
            await update.message.reply_text('Ничего не найдено на изображении.')
            await update.message.reply_photo(photo=InputFile(BytesIO(img_bytes), filename='orig.jpg'))
            return

        counts = {}
        best_examples = {}
        for d in det_list:
            cls = d['class']
            counts[cls] = counts.get(cls, 0) + 1
            if cls not in best_examples or d['confidence'] > best_examples[cls]['confidence']:
                best_examples[cls] = d

        lines = [f'Найдено объектов: {sum(counts.values())}']
        for cls, cnt in counts.items():
            lines.append(f'- {cls}: {cnt} (лучш. доверие {best_examples[cls]["confidence"]:.2f})')
        report = '\n'.join(lines)

        await update.message.reply_text(report)
        await update.message.reply_photo(photo=InputFile(BytesIO(annotated_bytes), filename='annotated.jpg'))

    except Exception as e:
        logger.exception('Ошибка при обработке фото:')
        await update.message.reply_text(f'Ошибка: {e}')

    # …остальная часть отправки фото и отчета…


# ---------------------------------------------------------------------------
# Вспомогательные функции: генерация текста через RuT5

async def ensure_text_model_loaded():
    """Ленивая загрузка RuT5 cointegrated/rut5-base-multitask."""
    global _text_tokenizer, _text_model, _text_pipeline
    if USE_HF_INFERENCE:
        return
    if _text_model is not None and _text_tokenizer is not None and _text_pipeline is not None:
        return
    async with _model_lock:
        if _text_model is not None and _text_tokenizer is not None and _text_pipeline is not None:
            return
        logger.info('Загружаю текстовую модель RuT5 (cointegrated/rut5-base-multitask)...')
        model_name = MODEL_NAME
        _text_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _text_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _text_model.to(DEVICE)
        _text_model.eval()
        _text_pipeline = pipeline(
            "text2text-generation",
            model=_text_model,
            tokenizer=_text_tokenizer,
            device=0 if DEVICE=='cuda' else -1
        )
        logger.info('Модель RuT5 загружена!')


def generate_text_local(prompt: str,
                        max_new_tokens: int = MAX_GEN_TOKENS,
                        temperature: float = TEMPERATURE,
                        top_p: float = TOP_P,
                        top_k: int = TOP_K,
                        num_return_sequences: int = NUM_RETURN_SEQUENCES) -> str:
    global _text_pipeline
    if _text_pipeline is None:
        raise RuntimeError('Локальная текстовая модель не загружена. Вызовите ensure_text_model_loaded перед этим.')

    with torch.no_grad():  # <-- ускорение
        output = _text_pipeline(
            prompt,
            max_length=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences
        )
    gen_text = output[0]["generated_text"]
    return gen_text.strip()


def generate_text_hf(prompt: str,
                     max_new_tokens: int = MAX_GEN_TOKENS,
                     temperature: float = TEMPERATURE,
                     top_p: float = TOP_P,
                     top_k: int = TOP_K,
                     num_return_sequences: int = NUM_RETURN_SEQUENCES) -> str:
    """Генерация через Hugging Face Inference API."""
    if not HF_API_TOKEN:
        raise RuntimeError('HF_API_TOKEN не задан, но USE_HF_INFERENCE=True')
    api_url = f'https://api-inference.huggingface.co/models/{MODEL_NAME}'
    headers = {'Authorization': f'Bearer {HF_API_TOKEN}'}
    payload = {
        'inputs': prompt,
        'parameters': {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'do_sample': True,
            'num_return_sequences': num_return_sequences
        }
    }
    logger.info('Вызов Hugging Face Inference API...')
    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f'Hugging Face API error: {resp.status_code} {resp.text}')
    result = resp.json()
    if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
        gen = result[0]['generated_text']
        return gen.strip()
    return str(result)


# ---------------------------------------------------------------------------
# YOLOv5 функции (не менялись)
async def ensure_yolo_loaded():
    global _yolo_model
    if _yolo_model is not None:
        return
    async with _model_lock:
        if _yolo_model is not None:
            return
        logger.info('Загружаю YOLOv5 модель...')
        try:
            if YOLO_WEIGHTS:
                logger.info(f'Загрузка веса YOLOv5 из {YOLO_WEIGHTS}...')
                _yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_WEIGHTS, force_reload=False)
            else:
                _yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            _yolo_model.to(DEVICE)
            logger.info('YOLOv5 загружен.')
        except Exception as e:
            logger.exception('Не удалось загрузить YOLOv5.')
            raise


def annotate_image_with_boxes(image: Image.Image, detections) -> Tuple[bytes, List[Dict]]:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', size=16)
    except Exception:
        font = ImageFont.load_default()
    det_list = []
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])
        conf = float(row['confidence'])
        cls_name = str(row['name'])
        label = f'{cls_name} {conf:.2f}'
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
        text_size = draw.textsize(label, font=font)
        draw.rectangle([xmin, ymin - text_size[1] - 4, xmin + text_size[0] + 4, ymin], fill='red')
        draw.text((xmin + 2, ymin - text_size[1] - 2), label, fill='white', font=font)
        det_list.append({'class': cls_name, 'confidence': float(conf), 'box': [xmin, ymin, xmax, ymax]})
    out = BytesIO()
    img.save(out, format='JPEG', quality=85)
    out.seek(0)
    return out.read(), det_list


def detect_objects_from_bytes(image_bytes: bytes) -> Tuple[List[Dict], bytes]:
    global _yolo_model
    if _yolo_model is None:
        raise RuntimeError('YOLO модель не загружена. Вызовите ensure_yolo_loaded.')
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    results = _yolo_model(np.array(img))
    try:
        df = results.pandas().xyxy[0]
    except Exception:
        df = None
    if df is None or df.empty:
        out = BytesIO()
        img.save(out, format='JPEG', quality=85)
        out.seek(0)
        return [], out.read()
    annotated_bytes, det_list = annotate_image_with_boxes(img, df)
    return det_list, annotated_bytes


# ---------------------------------------------------------------------------
# Telegram handlers

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Привет! Я бот для генерации текста (RuT5) и распознавания объектов (YOLOv5).\n\n"
        "Отправь текст — я сгенерирую продолжение.\n"
        "Отправь фото — я верну аннотированное изображение и список объектов.\n\n"
        "/start — показать это сообщение\n"
        "/help — справка по параметрам\n"
    )
    await update.message.reply_text(text)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        f"Параметры генерации:\nMODEL_NAME={MODEL_NAME}\n"
        f"USE_HF_INFERENCE={USE_HF_INFERENCE}\n"
        f"MAX_GEN_TOKENS={MAX_GEN_TOKENS}, TEMPERATURE={TEMPERATURE}, TOP_P={TOP_P}, TOP_K={TOP_K}\n"
    )
    await update.message.reply_text(text)


async def on_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    now = time.time()
    last = USER_LAST_TS.get(user_id, 0)
    if now - last < MIN_INTERVAL_SECONDS:
        await update.message.reply_text('Пожалуйста, не присылайте запросы так часто.')
        return
    USER_LAST_TS[user_id] = now

    prompt = update.message.text.strip()
    if not prompt:
        await update.message.reply_text('Пустое сообщение — отправьте текст для генерации.')
        return

    await update.message.chat.send_action(action='typing')

    try:
        loop = asyncio.get_event_loop()
        if USE_HF_INFERENCE:
            gen_text = await loop.run_in_executor(None, generate_text_hf, prompt)
        else:
            await ensure_text_model_loaded()
            gen_text = await loop.run_in_executor(None, generate_text_local, prompt)

        if not gen_text:
            await update.message.reply_text('Не удалось сгенерировать текст.')
            return

        if len(gen_text) > 4000:
            gen_text = gen_text[:4000] + '...'

        await update.message.reply_text(gen_text)

    except Exception as e:
        logger.exception('Ошибка при генерации текста:')
        await update.message.reply_text(f'Ошибка: {e}')


async def on_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    now = time.time()
    last = USER_LAST_TS.get(user_id, 0)
    if now - last < MIN_INTERVAL_SECONDS:
        await update.message.reply_text('Пожалуйста, не присылайте запросы так часто.')
        return
    USER_LAST_TS[user_id] = now

    photos = update.message.photo
    if not photos:
        await update.message.reply_text('Я не нашел фото в сообщении.')
        return

    photo = photos[-1]
    f = await photo.get_file()
    bio = BytesIO()
    await f.download(out=bio)
    bio.seek(0)
    img_bytes = bio.read()

    await update.message.chat.send_action(action='upload_photo')

    try:
        await ensure_yolo_loaded()
        loop = asyncio.get_event_loop()
        det_list, annotated_bytes = await loop.run_in_executor(None, detect_objects_from_bytes, img_bytes)

        if not det_list:
            await update.message.reply_text('Ничего не найдено на изображении.')
            await update.message.reply_photo(photo=InputFile(BytesIO(img_bytes), filename='orig.jpg'))
            return

        counts = {}
        best_examples = {}
        for d in det_list:
            cls = d['class']
            counts[cls] = counts.get(cls, 0) + 1
            if cls not in best_examples or d['confidence'] > best_examples[cls]['confidence']:
                best_examples[cls] = d

        lines = [f'Найдено объектов: {sum(counts.values())}']
        for cls, cnt in counts.items():
            lines.append(f'- {cls}: {cnt} (лучш. доверие {best_examples[cls]["confidence"]:.2f})')
        report = '\n'.join(lines)

        await update.message.reply_text(report)
        await update.message.reply_photo(photo=InputFile(BytesIO(annotated_bytes), filename='annotated.jpg'))

    except Exception as e:
        logger.exception('Ошибка при обработке фото:')
        await update.message.reply_text(f'Ошибка: {e}')


# ---------------------------------------------------------------------------
# Main
def build_application():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('help', cmd_help))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_message))
    return app


if __name__ == '__main__':
    logger.info('Старт приложения...')
    app = build_application()
    app.run_polling()
