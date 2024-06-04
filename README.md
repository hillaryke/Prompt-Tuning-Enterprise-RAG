# Precision RAG: Prompt Tuning For Building Enterprise Grade RAG Systems

# Scalable Data Warehouse for LLM Finetuning: API Design for High Throughput Data Ingestion and RAG Retrieval

## Introduction

This project introduces a powerful Retrieval-Augmented Generation (RAG) system, capable of reading books and extracting context from them. Leveraging the power of large language models, our system can understand and generate responses based on the content of a wide range of books. This makes it an invaluable tool for tasks such as question answering, summarization, and context-aware recommendations.


## Features

- **Book Reading Capability**: Book Reading Capability: Our RAG system can read and understand the content of books. This allows it to answer questions and generate responses based on the information contained in the books.
- **Context Extraction**: Not only can our system read books, but it can also understand the context in which information is presented. This enables it to provide more accurate and contextually relevant responses.

## Requirements Before Installation
  - Docker: This is used for creating, deploying, and running applications by using containers.
  - Docker Compose: This is a tool for defining and managing multi-container Docker applications.
  - Python: The project requires a Python version ~= 3.10.12

### Setup and Installation
1. **Clone the Repository**
    ```bash
    git clone git@github.com:hillaryke/Prompt-Tuning-Enterprise-RAG.git
    cd Prompt-Tuning-Enterprise-RAG
    ```

2. **Create a Virtual Environment and Install Dependencies**
```bash
python3.10 -m venv venv
source venv/bin/activate  # For Unix or MacOS
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```

4. **Environment Variables**
    - Create a `.env` file in the root directory and add the following environment variables:


## Testing
- To run the tests, execute the following command:
    ```bash
    make test
    ```
- This will run the tests using the `pytest` framework on the files in the `tests`  and also files with the naming convention `test_*.py` in project directory.
- The Api tests are written to test the endpoints of the FastAPI application.
    
## Conclusion
Our Retrieval-Augmented Generation (RAG) system is a powerful tool for extracting context and generating responses based on the content of books. By leveraging the capabilities of large language models, it provides a unique approach to tasks such as question answering, summarization, and context-aware recommendations. Whether you're a developer looking to integrate advanced language understanding into your application, or a researcher exploring the frontiers of natural language processing, our RAG system offers a robust and scalable solution. Start leveraging its features in your workflow today.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.