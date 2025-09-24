from chat_pipeline import run_pipeline
from flask_sqlalchemy import SQLAlchemy
import dotenv

# Загружаем переменные окружения из файла .env
env = dotenv.dotenv_values(".env")

# Инициализируем SQLAlchemy для работы с базой данных через Flask
db = SQLAlchemy()


class ChatHistory(db.Model):
    """
    Модель SQLAlchemy для хранения истории общения пользователя с LLM.

    Атрибуты:
        id (int): Уникальный идентификатор записи.
        user_message (str): Сообщение пользователя.
        llm_reply (str): Ответ языковой модели.
        timestamp (datetime): Время создания записи.
    """
    id = db.Column(db.Integer, primary_key=True)  # Уникальный идентификатор
    user_message = db.Column(db.Text, nullable=False)  # Текст сообщения пользователя
    llm_reply = db.Column(db.Text, nullable=False)  # Ответ модели
    timestamp = db.Column(db.DateTime, server_default=db.func.now())  # Временная метка создания записи


def format_msg_with_history(user_message, history, depth=2):
    """
    Форматирует сообщение пользователя вместе с ограниченной историей чата.

    Аргументы:
        user_message (str): Текст сообщения пользователя.
        history (list of dict): История сообщений в формате [{role: str, content: str}, ...].
        depth (int): Количество последних сообщений из истории, которые нужно включить в отформатированное сообщение. По умолчанию 2.

    Возвращает:
        str: Отформатированное сообщение, которое включает историю чата и текущее сообщение пользователя.
    """
    out = ''
    for msg in history[:-depth]:
        content = msg['content']
        if msg['role'] == 'user':
            out += f'\\user:\n """  {content} \n """ \n '
        elif msg['role'] == 'assistant':
            out += f'\\assistant:\n """ {content} \n """ \n'
    out += f"Пользователь спрашивает: {user_message}"
    return out


def chat_with_llm(user_message, history=None):
    """
    Чат с использованием сервиса LLM.

    Аргументы:
        user_message (str): Сообщение пользователя.

    Возвращает:
        str: Ответ LLM.
    """
    if not history is None:
        user_message_with_history = format_msg_with_history(user_message, history)
        print('user_message_with_history = ', user_message_with_history)
        llm_response = run_pipeline(user_message_with_history)
    else:
        llm_response = run_pipeline(f"Пользователь спрашивает: {user_message}")
    
    if not history is None:
        history.append({"role": "user", "content": user_message})  # добавляем сообщение пользователя в историю
        history.append({"role": "assistant", "content": llm_response})
    
    return llm_response
