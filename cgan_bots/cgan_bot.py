from io import BytesIO
from typing import *

import numpy as np
import torch
import torchvision
from telegram.ext import (Updater,
                          CommandHandler,
                          Dispatcher,
                          CallbackContext,
                          CallbackQueryHandler)
from telegram.update import Update
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


class CGANBot:
    def __init__(self, bot_key: str, model, input_shape: Tuple[int, int, int, int], label_dict: Dict[str, int]):
        self.label_dict = label_dict
        self.input_shape = input_shape
        self.model = model
        self.__updater: Updater = Updater(bot_key, use_context=True)
        self.__dispatcher: Dispatcher = self.__updater.dispatcher
        self.__dispatcher.add_handler(CommandHandler("classes", self.get_classes))
        self.__dispatcher.add_handler(CommandHandler("generate", self.gen_image))
        self.__dispatcher.add_handler(CallbackQueryHandler(self.keyboard_callback))

    def get_classes(self, update: Update, context: CallbackContext):
        answer = ", ".join([str(k) for k in self.label_dict.keys()])
        update.message.reply_text(answer, quote=True)

    def gen_image(self, update: Update, context: CallbackContext):
        labels = update.message.text.split(" ")
        if len(labels) < 2:
            keyboard = []
            for k in self.label_dict.keys():
                keyboard.append([InlineKeyboardButton(k, callback_data=k)])
            update.message.reply_text("Choose a label", reply_markup=InlineKeyboardMarkup(keyboard), quote=True)
            return
        label = labels[1]
        if label not in self.label_dict.keys():
            update.message.reply_text("unknown class. use /classes to list all supported class labels")
            return
        label = self.label_dict[label]
        bio = self.generate_image(label)
        update.message.reply_photo(bio, quote=True)

    def keyboard_callback(self, update: Update, context: CallbackContext):
        query = update.callback_query
        label = query.data
        if label not in self.label_dict.keys():
            query.edit_message_text(text="...my programmer is apparently not able to do his job.... (╯°□°)╯︵ ┻━┻ ")
            return
        ilabel = self.label_dict[label]
        bio = self.generate_image(ilabel)
        query.edit_message_text(text=f"I proudly present to you a real generated person. Flavour: {label}")
        query.message.reply_photo(bio, quote=True)

    def generate_image(self, label) -> BytesIO:
        noise = torch.tensor(np.random.normal(0, 1, self.input_shape), dtype=torch.float)
        image = self.model(noise, [label])
        image = (image + 1) / 2
        image = torchvision.transforms.ToPILImage()(image[0])
        bio = BytesIO()
        bio.name = 'image.jpeg'
        image.save(bio, 'JPEG')
        bio.seek(0)
        return bio

    def run(self):
        """
        start bot
        """
        # Start the Bot
        self.__updater.start_polling()

        # Run the bot until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        self.__updater.idle()
