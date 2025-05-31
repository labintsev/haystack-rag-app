"""
Tutorial 1. Extractive question answering pipeline
Практика 1. Извлечение информации из документов для ответа на заданный вопрос. 
"""
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.readers import ExtractiveReader

docs = [
    Document(content="Золото стоит 8 372,4100 рублей за грамм"),
    Document(content="Серебро стоит 84,3500 рублей за грамм"),
    Document(content="Платина стоит 2 750,0300 рублей за грамм"),
    Document(content="Палладий стоит 2 466,9300 рублей за грамм"),
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
