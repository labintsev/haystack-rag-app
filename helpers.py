def format_query(query):
    """Format the input query for processing."""
    return query.strip().lower()

def process_results(results):
    """Process the results from the retriever to a more usable format."""
    return [result['content'] for result in results]

def log_message(message):
    """Log messages for debugging purposes."""
    print(f"[LOG] {message}")