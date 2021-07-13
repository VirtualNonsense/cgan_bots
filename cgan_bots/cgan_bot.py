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
    def __init__(self,
                 bot_key: str,
                 people_model,
                 people_shape: Tuple[int, int, int, int],
                 people_dict: Dict[str, int],
                 art_model,
                 art_shape: Tuple[int, int, int, int],
                 art_dict: Dict[str, int],):
        self.people_dict = people_dict
        self.people_input_shape = people_shape
        self.people_model = people_model
        self.art_model = art_model
        self.art_shape = art_shape
        self.art_dict = art_dict
        self.__updater: Updater = Updater(bot_key,
                                          use_context=True,
                                          request_kwargs={'read_timeout': 6, 'connect_timeout': 7})
        self.__peopleCMD = "generate_people"
        self.__coverartCMD = "generate_coverart"
        self.__dispatcher: Dispatcher = self.__updater.dispatcher
        self.__dispatcher.add_handler(CommandHandler("help", self.help))
        self.__dispatcher.add_handler(CommandHandler("start", self.help))
        self.__dispatcher.add_handler(CommandHandler(self.__peopleCMD, self.gen_face))
        self.__dispatcher.add_handler(CommandHandler(self.__coverartCMD, self.gen_artwork))
        self.__dispatcher.add_handler(CallbackQueryHandler(self.keyboard_callback))

    def help(self, update: Update, context: CallbackContext):
        return update.message.reply_text("Use /generate_people or /generate_people [class] "
                                         "to generate a human face with a specific trade.\n"
                                         "Use /generate_coverart or /generate_coverart [genre] "
                                         "to generate cover artwork for a given genre.", quote=True)

    def gen_artwork(self, update: Update, context: CallbackContext):
        message = update.message if update.message is not None else update.edited_message

        if message.text.find(" ") == -1:
            message.reply_text("Choose a label",
                               reply_markup=CGANBot.generate_keyboard(self.art_dict, self.__coverartCMD), quote=True)
            return

        labels = message.text.split(" ")
        if len(labels) < 2:
            message.reply_text("Choose a label",
                               reply_markup=CGANBot.generate_keyboard(self.art_dict, self.__coverartCMD), quote=True)
        label = labels[1].capitalize()
        if label not in self.art_dict.keys():
            message.reply_text(f"The class \'{label}\' is not supported. Please choose a class from below",
                               reply_markup=CGANBot.generate_keyboard(self.art_dict, self.__coverartCMD),
                               quote=True)
            return
        label = self.art_dict[label]
        bio = self.generate_artwork(label)
        message.reply_photo(bio, quote=True)

    def gen_face(self, update: Update, context: CallbackContext):
        message = update.message if update.message is not None else update.edited_message

        if message.text.find(" ") == -1:
            message.reply_text("Choose a label",
                               reply_markup=CGANBot.generate_keyboard(self.people_dict, self.__peopleCMD),
                               quote=True)
            return

        labels = message.text.split(" ")
        if len(labels) < 2:
            message.reply_text("Choose a label",
                               reply_markup=CGANBot.generate_keyboard(self.people_dict, self.__peopleCMD),
                               quote=True)
        label = labels[1].capitalize()
        if label not in self.people_dict.keys():
            message.reply_text(f"The class \'{label}\' is not supported. Please choose a class from below",
                               reply_markup=CGANBot.generate_keyboard(self.people_dict, self.__peopleCMD),
                               quote=True)
            return
        label = self.people_dict[label]
        bio = self.generate_human_face(label)
        message.reply_photo(bio, quote=True)

    @staticmethod
    def generate_keyboard(dict: Dict[str, int], command: str):
        keyboard = []
        for k in dict.keys():
            keyboard.append([InlineKeyboardButton(k, callback_data=",".join([command, k]))])
        return InlineKeyboardMarkup(keyboard)

    def keyboard_callback(self, update: Update, context: CallbackContext):
        query = update.callback_query
        command, label = query.data.split(",")
        bio: BytesIO
        if command == self.__coverartCMD:
            if label not in self.art_dict.keys():
                query.edit_message_text(text=f"...my programmer is apparently not able to do his job.... (╯°□°)╯︵ ┻━┻\n"
                                             f"He asked me to find {label} in {self.art_dict.keys()}.... \n"
                                             f"Clearly a n00b.")
                return
            ilabel = self.art_dict[label]
            bio = self.generate_artwork(ilabel)
            query.edit_message_text(text=f"I proudly present to you a groovy piece of {label.lower()} cover artwork.")
        elif command == self.__peopleCMD:
            if label not in self.people_dict.keys():
                query.edit_message_text(text=f"...my programmer is apparently not able to do his job.... (╯°□°)╯︵ ┻━┻\n"
                                             f"He asked me to find {label} in {self.people_dict.keys()}.... \n"
                                             f"Clearly a n00b.")
                return
            ilabel = self.people_dict[label]
            bio = self.generate_human_face(ilabel)
            query.edit_message_text(text=f"I proudly present to you a real generated person. Flavour: {label}")
        else:
            query.edit_message_text(text="...my programmer is apparently not able to do his job.... (╯°□°)╯︵ ┻━┻ \n"
                                         f"He asked me to {command} even though I don't know how....")
            return
        query.message.reply_photo(bio, quote=True)

    def generate_human_face(self, label) -> BytesIO:
        noise = torch.tensor(np.random.normal(0, 1, self.people_input_shape), dtype=torch.float)
        image = self.people_model(noise, [label])
        image = (image + 1) / 2
        image = torchvision.transforms.ToPILImage()(image[0])
        bio = BytesIO()
        bio.name = 'face.png'
        image.save(bio, 'PNG')
        bio.seek(0)
        return bio

    def generate_artwork(self, label) -> BytesIO:
        noise = torch.tensor(np.random.normal(0, 1, self.art_shape), dtype=torch.float)
        image = self.art_model(noise, [label])
        image = (image + 1) / 2
        image = torchvision.transforms.ToPILImage()(image[0])
        bio = BytesIO()
        bio.name = 'artwork.png'
        image.save(bio, 'PNG')
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
