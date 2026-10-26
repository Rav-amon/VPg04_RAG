"""Telegram-интерфейс RAG-агента на PyTelegramBotAPI."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import telebot
from dotenv import load_dotenv

from rag_agent import (
  ENV_PATH,
  RagAgent,
  RagAgentError,
  extract_urls,
  has_question,
  remaining_text,
)

logger = logging.getLogger(__name__)

TELEGRAM_CHUNK_SIZE = 4000
POLLING_TIMEOUT_SECONDS = 30
MAX_URLS_PER_COMMAND = 3
HELP_TEXT = (
  'Я помощник с памятью и векторной базой.\n'
  'Перед ответом ищу фрагменты в Pinecone.\n'
  'Фразы вроде «меня зовут…» или «запомни, что…» сохраняю как эмбеддинги.\n'
  '/add текст — сразу записывает новые данные в базу знаний.\n'
  '/read https://пример.com — читает HTML, режет страницу на фрагменты '
  'и сохраняет эмбеддинги.\n'
  'После ссылки можно задать вопрос, и я отвечу уже по сохранённой странице.\n'
  'Вопрос про погоду или ссылка на открытый API запускает GET, '
  'и ответ строится по телу ответа.\n'
  'Команды: /start, /help, /add, /read, /cancel.'
)


def load_telegram_token() -> str:
  """Читает токен бота из окружения, не печатая его."""
  load_dotenv(ENV_PATH)
  token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
  if not token:
    raise RagAgentError(
      code='config_missing',
      message='Не задана переменная окружения TELEGRAM_BOT_TOKEN.',
      details={'variable': 'TELEGRAM_BOT_TOKEN'},
    )
  return token


def split_telegram_text(
  text: str,
  limit: int = TELEGRAM_CHUNK_SIZE,
) -> list[str]:
  """Делит длинный ответ на части, которые принимает Telegram."""
  rest = text.strip() or 'Пустой ответ.'
  chunks: list[str] = []
  while rest:
    if len(rest) <= limit:
      chunks.append(rest)
      break
    cut = rest.rfind('\n', 0, limit)
    if cut < limit // 2:
      cut = limit
    piece = rest[:cut].strip()
    if piece:
      chunks.append(piece)
    rest = rest[cut:].strip()
  return chunks or ['Пустой ответ.']


def build_bot(token: str, agent: RagAgent) -> telebot.TeleBot:
  """Собирает бота, который отвечает через общий экземпляр RagAgent."""
  bot = telebot.TeleBot(token)

  @bot.message_handler(commands=['start', 'help'])
  def on_start(message: telebot.types.Message) -> None:
    _send_reply(bot, message, HELP_TEXT)

  @bot.message_handler(commands=['cancel'])
  def on_cancel(message: telebot.types.Message) -> None:
    bot.clear_step_handler_by_chat_id(message.chat.id)
    _send_reply(bot, message, 'Ожидание отменено.')

  @bot.message_handler(commands=['add'])
  def on_add(message: telebot.types.Message) -> None:
    payload = _knowledge_payload(message)
    if not payload:
      _send_reply(
        bot,
        message,
        'Пришлите текст следующим сообщением или так: /add новый факт.',
      )
      bot.register_next_step_handler(message, _save_following_text)
      return
    _reply_with_action(
      bot,
      message,
      lambda: _store_knowledge(agent, message, payload),
    )

  @bot.message_handler(commands=['read'])
  def on_read(message: telebot.types.Message) -> None:
    urls = _message_urls(message)
    if not urls:
      _send_reply(bot, message, 'Пришлите ссылку http или https.')
      bot.register_next_step_handler(message, _read_following_text)
      return
    question = _question_besides_urls(message)
    _reply_with_action(
      bot,
      message,
      lambda: _read_pages(agent, message, urls, question),
    )

  @bot.message_handler(content_types=['text'])
  def on_text(message: telebot.types.Message) -> None:
    text = (message.text or '').strip()
    if not text or text.startswith('/'):
      if text.startswith('/'):
        _send_reply(bot, message, 'Такой команды нет. Напишите /help.')
      return
    _reply_with_action(
      bot,
      message,
      lambda: agent.answer(text, user_id=_user_id(message)),
    )

  try:
    bot.set_my_commands([
      telebot.types.BotCommand('start', 'Что умеет помощник'),
      telebot.types.BotCommand('help', 'Список команд'),
      telebot.types.BotCommand('add', 'Добавить текст в базу знаний'),
      telebot.types.BotCommand('read', 'Прочитать HTML-страницу по ссылке'),
      telebot.types.BotCommand('cancel', 'Отменить ожидание текста или ссылки'),
    ])
  except Exception:
    logger.exception('Не удалось опубликовать меню команд')

  def _save_following_text(next_message: telebot.types.Message) -> None:
    text = (next_message.text or '').strip()
    if _is_cancel(text):
      _send_reply(bot, next_message, 'Добавление отменено.')
      return
    if not text or text.startswith('/'):
      _send_reply(bot, next_message, 'Нужен текст. Пример: /add новый факт.')
      return
    _reply_with_action(
      bot,
      next_message,
      lambda: _store_knowledge(agent, next_message, text),
    )

  def _read_following_text(next_message: telebot.types.Message) -> None:
    text = (next_message.text or '').strip()
    if _is_cancel(text):
      _send_reply(bot, next_message, 'Чтение страницы отменено.')
      return
    urls = extract_urls(text)
    if not urls:
      _send_reply(bot, next_message, 'В сообщении нет ссылки http или https.')
      return
    question = remaining_text(text, urls)
    _reply_with_action(
      bot,
      next_message,
      lambda: _read_pages(agent, next_message, urls, question),
    )

  return bot


def main() -> None:
  """Запускает длинный опрос Telegram и отвечает через RAG-агента."""
  logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(name)s: %(message)s',
  )
  try:
    token = load_telegram_token()
    agent = RagAgent()
    bot = build_bot(token, agent)
  except RagAgentError as error:
    print(error.message)
    raise SystemExit(1) from error
  logger.info('Telegram-бот запущен')
  bot.infinity_polling(
    skip_pending=True,
    timeout=POLLING_TIMEOUT_SECONDS,
    long_polling_timeout=POLLING_TIMEOUT_SECONDS,
  )


def _knowledge_payload(message: telebot.types.Message) -> str:
  argument = _command_argument(message)
  if argument:
    return argument
  reply = message.reply_to_message
  if reply is None:
    return ''
  return (reply.text or '').strip()


def _message_urls(message: telebot.types.Message) -> list[str]:
  parts = [_command_argument(message)]
  reply = message.reply_to_message
  if reply is not None:
    parts.append(reply.text or '')
    parts.extend(_text_link_urls(reply.entities))
  parts.extend(_text_link_urls(message.entities))
  urls: list[str] = []
  for part in parts:
    for url in extract_urls(part):
      if url not in urls:
        urls.append(url)
  return urls[:MAX_URLS_PER_COMMAND]


def _text_link_urls(entities: list[telebot.types.MessageEntity] | None) -> list[str]:
  if not entities:
    return []
  return [
    entity.url
    for entity in entities
    if entity.type == 'text_link' and entity.url
  ]


def _question_besides_urls(message: telebot.types.Message) -> str:
  argument = _command_argument(message)
  return remaining_text(argument, extract_urls(argument))


def _command_argument(message: telebot.types.Message) -> str:
  argument = telebot.util.extract_arguments(message.text or '')
  if not argument:
    return ''
  return argument.strip()


def _is_cancel(text: str) -> bool:
  command = text.split(maxsplit=1)[0].split('@', maxsplit=1)[0].lower()
  return command == '/cancel'


def _store_knowledge(agent: RagAgent, message: telebot.types.Message, text: str) -> str:
  try:
    count = agent.add_knowledge(text, user_id=_user_id(message))
  except RagAgentError as error:
    logger.error(
      'Не удалось добавить данные: %s %s',
      error.code,
      error.message,
    )
    return error.message
  return f'В базу знаний добавлено фрагментов: {count}.'


def _read_pages(
  agent: RagAgent,
  message: telebot.types.Message,
  urls: list[str],
  question: str,
) -> str:
  lines = ['Страница прочитана и сохранена в векторную базу.']
  try:
    for url in urls:
      count = agent.ingest_url(url)
      lines.append(f'{url}: фрагментов {count}.')
  except RagAgentError as error:
    logger.error('Не удалось прочитать страницу: %s', error.code)
    return error.message
  if not has_question(question):
    lines.append('Отдельного вопроса в сообщении не было.')
    return '\n'.join(lines)
  answer = agent.answer(question, user_id=_user_id(message))
  return '\n'.join(lines) + '\n\n' + answer


def _reply_with_action(
  bot: telebot.TeleBot,
  message: telebot.types.Message,
  action: Callable[[], str],
) -> None:
  bot.send_chat_action(message.chat.id, 'typing')
  try:
    reply = action()
  except Exception:
    logger.exception('Сбой обработки сообщения чата %s', message.chat.id)
    reply = 'Не получилось обработать сообщение. Попробуйте ещё раз.'
  _send_reply(bot, message, reply)


def _user_id(message: telebot.types.Message) -> str:
  if message.from_user is None:
    return ''
  return str(message.from_user.id)


def _send_reply(
  bot: telebot.TeleBot,
  message: telebot.types.Message,
  text: str,
) -> None:
  chunks = split_telegram_text(text)
  bot.reply_to(message, chunks[0])
  for chunk in chunks[1:]:
    bot.send_message(message.chat.id, chunk)


if __name__ == '__main__':
  main()
