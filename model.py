from chat_pipeline import run_pipeline


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

    history.append({"role": "user", "content": user_message})  # добавляем сообщение пользователя в историю
    history.append({"role": "assistant", "content": llm_response})
    return llm_response
