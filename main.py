# Entry point for the RAG system
from pprint import pprint
from extractive_qa_pipeline import run_pipeline

def main(query):
    # Run the RAG pipeline
    response = run_pipeline(query)

    # Print the generated response
    pprint(response)

if __name__ == "__main__":
    query = "Сколько стоит золото?"
    main(query)
