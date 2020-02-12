# ------------------------------------------------------------------------------
# A Bot to send updates over Telegram. Fill in 'chat_id' and 'token' to use.
# ------------------------------------------------------------------------------

import telegram as tel

class TelegramBot():
    def __init__(self, chat_id='', token=''):
        self.chat_id = chat_id
        self.bot = tel.Bot(token=token)

    def send_msg(self, msg):
        self.bot.send_message(chat_id=self.chat_id, text=msg)
