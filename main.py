import logging
import cgan_bots.cgan_bot
from secrets import bot_key

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    b = cgan_bots.cgan_bot.CGANBot(bot_key)
    b.run()

