# Legal Document Assistant – RAG Based Question Answering

## Project Overview

The Legal Document Assistant is an AI-powered application that uses **Retrieval-Augmented Generation (RAG)** to understand legal documents, generate summaries, and answer questions based on the document's content.

The application combines document processing, text chunking, semantic search, vector similarity retrieval, and Large Language Models (LLMs) to provide context-aware responses from uploaded legal documents.

## Key Features

- **Legal Document Summarization:** Generate concise summaries of legal documents using LLM-based summarization.
- **Document Question Answering:** Ask questions and receive answers based on the uploaded document.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document sections before generating answers.
- **Semantic Search:** Identify relevant content using sentence embeddings and vector similarity.
- **FAISS Vector Search:** Store and retrieve document embeddings efficiently.
- **TF-IDF Fallback Retrieval:** Use TF-IDF and cosine similarity when vector-based retrieval is unavailable.
- **Context-Grounded Responses:** Generate answers using retrieved document content to reduce unsupported responses.
- **PDF Processing:** Extract and preprocess text from PDF documents.
- **Conversation History:** Maintain chat conversations and previous messages.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| NLP | NLTK, Transformers |
| LLM | Ollama |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| Retrieval | TF-IDF, Cosine Similarity |
| Backend Framework | Flask |
| PDF Processing | PyPDF2 |
| Database | SQLite |

## RAG Architecture

The application follows a Retrieval-Augmented Generation workflow.

1. **Document Upload:** The user provides a legal document in PDF format.
2. **Text Extraction:** Extract textual content from the uploaded document.
3. **Text Preprocessing:** Clean extracted text and normalize formatting.
4. **Document Chunking:** Divide the document into smaller, meaningful sections while preserving legal clause boundaries where possible.
5. **Embedding Generation:** Generate semantic embeddings using Sentence Transformers.
6. **Vector Indexing:** Store document embeddings in a FAISS vector index.
7. **Semantic Retrieval:** Retrieve relevant document sections based on the user's question.
8. **LLM Generation:** Pass the retrieved context and question to the LLM.
9. **Response Generation:** Generate a context-grounded answer using the retrieved document content.

## Retrieval Mechanism

The project implements two retrieval approaches:

**1. FAISS-Based Vector Retrieval**
- Generate document embeddings using Sentence Transformers.
- Create a FAISS similarity search index.
- Retrieve the most relevant document chunks based on semantic similarity.

**2. TF-IDF-Based Fallback Retrieval**
- Convert document chunks into TF-IDF representations.
- Calculate cosine similarity between the question and document chunks.
- Retrieve relevant sections when FAISS-based retrieval is unavailable.

## Project Structure

```text
Legal-Document-RAG/
│
├── main.py
├── rag.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sivanagasaibabu2002-hash/Legal-Document-RAG.git
cd Legal-Document-RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

Windows:
```bash
venv\Scripts\activate
```

Linux / macOS:
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

Install Ollama and download the required models

Make sure the Ollama service is running before starting the application.

### 5. Run the Application

## Important Notes

- The application uses locally hosted Ollama models for LLM-based generation.
- The first execution may require downloading transformer models and embedding models.
- FAISS-based retrieval requires the corresponding embedding dependencies.
- This repository shares the core Python implementation. Frontend templates, static assets, sample documents, and local database files are intentionally excluded.

## Future Improvements

- Improve retrieval accuracy using advanced embedding models.
- Implement hybrid retrieval combining semantic and keyword-based search.
- Add document citation and source-reference highlighting.
- Optimize document processing for large legal documents.
- Explore additional LLMs for improved response quality.

## Disclaimer

This project is intended for educational and technical demonstration purposes. Generated summaries and responses should not be considered professional legal advice.
