"""
Tutorial 2. Chat with hugging face with custom context
Практика 2. Чат с языковой моделью hugging face с заданным контекстом.
"""

from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

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

# Create document store and save documents with embeddings
document_store = InMemoryDocumentStore()
doc_embedder = SentenceTransformersDocumentEmbedder(model=embedder_model)
doc_embedder.warm_up()
docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

# Create another embedder for the user query
text_embedder = SentenceTransformersTextEmbedder(model=embedder_model)

# Create retriever, chat generator and prompt builder
retriever = InMemoryEmbeddingRetriever(document_store, top_k=3)
chat_generator = OllamaChatGenerator(
    model="gemma3:4b", url="http://127.0.0.1:11434"
)

prompt_builder = ChatPromptBuilder(template=template, required_variables=["question"])

# Make pipeline and add components
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

# Connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")


def run_pipeline(question):
    response = basic_rag_pipeline.run(
        {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}
    )

    return response["llm"]["replies"][0].text
