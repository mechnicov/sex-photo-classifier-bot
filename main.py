from dotenv import load_dotenv
load_dotenv()

import os
import logging
from lib.sex_photo_classifier import SexPhotoClassifier
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler

import requests
import json

logging.basicConfig(
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level = logging.INFO
)

logger = logging.getLogger(__name__)

sex_photo_classifier = SexPhotoClassifier()

def start(update, context):
    update.message.reply_text('Hi! Send me photo of human. I will try to guess if it is a man or a woman')

def main():
    updater = Updater(os.getenv('BOT_TOKEN'), use_context = True)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('help', start))
    dp.add_handler(MessageHandler(Filters.photo, process))
    dp.add_handler(MessageHandler(Filters.document.category('image'), process))
    dp.add_handler(CallbackQueryHandler(feedback))

    updater.start_polling()
    updater.idle()

def process(update, context):
    photo = update.message.photo[-1] if update.message.photo else update.message.document

    photo_file = photo.get_file()
    image_path = photo_file.file_path

    sex = sex_photo_classifier.classify(image_path)

    update.message.reply_text(f'Probably it is a {sex}')

    image_path = __shorten_url(image_path)

    decision = f'[{sex}] {image_path}'

    logger.info(decision)

    keyboard = [
        [
            InlineKeyboardButton('👍', callback_data = f'👍 {decision}'),
            InlineKeyboardButton('👎', callback_data = f'👎 {decision}'),
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Is it right?', reply_markup = reply_markup)

def feedback(update, context):
    query = update.callback_query
    query.answer()

    feedback = query.data[0]

    query.edit_message_text(text = f'Your feedback: {feedback}')

    logger.info(query.data)

def __shorten_url(url):
    params = {
      'destination': url,
    }

    headers = {
      'Content-type': 'application/json',
      'apikey': os.getenv('REBRANDLY_API_TOKEN'),
    }

    response = requests.post(
        'https://api.rebrandly.com/v1/links',
        data = json.dumps(params),
        headers = headers
    ).json()

    return f'https://{response["shortUrl"]}'

if __name__ == '__main__':
    main()
