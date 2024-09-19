# RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot that uses OpenAI's GPT model to provide relevant answers to user queries by referencing the content from uploaded PDF files. It utilizes Pinecone for storing and querying document embeddings to retrieve relevant information.

## Features

- **PDF Ingestion**: Upload PDF files to extract text and store the embeddings in Pinecone.
- **Pinecone Indexing**: The Pinecone vector database is used to store and retrieve document embeddings.
- **OpenAI GPT-4**: GPT-4 is used to generate natural language answers based on the context from the PDF.
- **Streamlit UI**: A web-based interface built with Streamlit for easy interactions.
- **Real-time Chat Interface**: Ask questions based on the uploaded documents and get relevant responses.

## Prerequisites

Before running this application, make sure you have:

- Python
- Pinecone account and API key
- OpenAI API key
- Docker (optional but recommended for containerized deployment)

## Setup

### 1. Clone the Repository

```bash
git clone git@github.com:HarshShahCitrusbug/tactical-edge-ai-test.git
cd tactical-edge-ai-test
```

### 2. Setup Environment Variables

Add the required environment variables mentioned in env.temp file.


### 3. Running the App Using Docker

* Build the Docker Image

To build the Docker image, run:

```bash
docker build -t rag-chatbot-app .
```

* Run the Docker Container

To run the Docker container, use:

```bash
docker run -p 8501:8501 rag-chatbot-app
```


### 3. Running the App Locally

To run the script locally use the following command:

```bash
    streamlit run app.py
```