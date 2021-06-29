from typing import *
from telegram.ext import (Updater, MessageHandler, CommandHandler, StringCommandHandler, Filters, Dispatcher)


class CGANBot:
    def __init__(self, bot_key: str):
        self.__updater: Updater = Updater(bot_key, use_context=True)
        self.__dispatcher: Dispatcher = self.__updater.dispatcher

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
