from haystack import Document

def format_query(query):
    """Format the input query for processing."""
    return query.strip().lower()

def process_results(results):
    """Process the results from the retriever to a more usable format."""
    return [result['content'] for result in results]

def log_message(message):
    """Log messages for debugging purposes."""
    print(f"[LOG] {message}")


def read_from_file(file_name):
    """Read documents from file, one line - one document """
    with open(file_name, encoding="utf-8") as f:
        lines = f.readlines()
    return [Document(content=line) for line in lines]

