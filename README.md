# Haystack RAG Application

This project implements a Retrieval-Augmented Generation (RAG) system using the Haystack framework. The RAG system combines the capabilities of a retriever and a generator to provide enhanced responses based on relevant documents.

## Project Structure

```
haystack-rag-app
├── src
│   ├── main.py          # Entry point for the application
│   ├── retriever
│   │   └── retriever.py # Contains the Retriever class for fetching documents
│   ├── generator
│   │   └── generator.py # Contains the Generator class for generating responses
│   ├── pipelines
│   │   └── rag_pipeline.py # Orchestrates the retrieval and generation process
│   └── utils
│       └── helpers.py   # Utility functions for common tasks
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/haystack-rag-app.git
   cd haystack-rag-app
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory and add any necessary API keys and configuration settings.

## Usage Guidelines

To run the RAG system, execute the following command:
```
python src/main.py
```

## RAG System Overview

The RAG system consists of two main components:

- **Retriever:** This component fetches relevant documents based on the input query. It is implemented in `src/retriever/retriever.py`.

- **Generator:** This component generates a response based on the retrieved documents. It is implemented in `src/generator/generator.py`.

The integration of these components is managed in `src/pipelines/rag_pipeline.py`, which orchestrates the entire process.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.