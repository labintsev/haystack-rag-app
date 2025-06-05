# Entry point for the RAG system
from pprint import pprint
from extractive_qa_pipeline import run_pipeline


if __name__ == "__main__":
    query = "Сколько стоит золото?"
    # Run the pipeline
    response = run_pipeline(query)

    # Print the generated response
    pprint(response)
