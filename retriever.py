class Retriever:
    def __init__(self, document_store):
        self.document_store = document_store

    def retrieve_documents(self, query):
        # Logic to retrieve documents based on the query
        relevant_documents = self.document_store.query(query)
        return relevant_documents

    def add_document(self, document):
        # Logic to add a document to the document store
        self.document_store.add(document)