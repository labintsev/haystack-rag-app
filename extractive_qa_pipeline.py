"""
Tutorial 1. Extractive question answering pipeline
Практика 1. Извлечение информации из документов для ответа на заданный вопрос. 
"""
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.readers import ExtractiveReader

docs = [
    Document(content="iPhone 15 Pro стоит 98 000 рублей"),
    Document(content="Samsung Galaxy S24 Ultra стоит 115 000 рублей"),
    Document(content="Xiaomi 14 Pro стоит 75 000 рублей"),
    Document(content="Google Pixel 9 Pro стоит 95 000 рублей"),
]

document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

retriever = InMemoryBM25Retriever(document_store=document_store)
reader = ExtractiveReader(model="deepset/roberta-base-squad2-distilled")

extractive_qa_pipeline = Pipeline()
extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
extractive_qa_pipeline.add_component(instance=reader, name="reader")
extractive_qa_pipeline.connect("retriever.documents", "reader.documents")


def run_pipeline(query):
    response = extractive_qa_pipeline.run(
        data={
            "retriever": {"query": query, "top_k": 3},
            "reader": {"query": query, "top_k": 2},
        }
    )
    return response["reader"]["answers"][0]
