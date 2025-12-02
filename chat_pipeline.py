"""
Tutorial 2. Chat with hugging face with custom context
Практика 2. Чат с языковой моделью hugging face с заданным контекстом.
"""

from haystack import Pipeline
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator

from dotenv import dotenv_values
from helpers import read_from_file

env = dotenv_values()

embedder_model = "sentence-transformers/all-MiniLM-L6-v2"

docs = read_from_file('data/test.txt')

template = [
    ChatMessage.from_user(
        """
Ты консультант по продаже сотовых телефонов. 
Ответь на вопрос: {{question}} 
Используй контекст:
{% for document in documents %}
    {{ document.content }}
{% endfor %}
"""
    )
]

# Создаем хранилище документов и сохраняем документы с эмбеддингами
document_store = InMemoryDocumentStore()
doc_embedder = SentenceTransformersDocumentEmbedder(model=embedder_model)
doc_embedder.warm_up()
docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

# Создаем другой эмбеддер для пользовательского запроса
text_embedder = SentenceTransformersTextEmbedder(model=embedder_model)

# Создаем ретривер, генератор чата и построитель подсказок
retriever = InMemoryEmbeddingRetriever(document_store)
chat_generator = OpenAIChatGenerator(
    api_key = Secret.from_token(env["YA_API_KEY"]),
    model = f"gpt://{env['YA_FOLDER_ID']}/yandexgpt-lite",
    api_base_url="https://llm.api.cloud.yandex.net/v1"
)

prompt_builder = ChatPromptBuilder(template=template, required_variables=["question"])

# Создаем конвейер и добавляем компоненты
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

# Соединяем компоненты друг с другом
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")


def run_pipeline(question):
    response = basic_rag_pipeline.run(
        {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}
    )

    return response["llm"]["replies"][0].text
