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

### Clone the Repository

```bash
git clone git@github.com:HarshShahCitrusbug/tactical-edge-ai-test.git
cd tactical-edge-ai-test
```


### Run the App 

**Using Docker**

1. Update the environment variables in the dockerfile

```bash
INDEX_NAME=<your-pinecone-index-name>
EMBEDDING_DIMENSION=<dimension-of-embeddings>
OPENAI_MODEL=<your-openai-model>
PINECONE_API_KEY=<your-pinecone-api-key>
OPENAI_API_KEY=<your-openai-api-key>
```


2. Build the Docker Image

To build the Docker image, run:

```bash
docker build -t rag-chatbot-app .
```

3. Run the Docker Container

To run the Docker container, use:

```bash
docker run -p 8501:8501 rag-chatbot-app
```


**Without Using Docker**

1. Create virtual environment

Create and activate the virtual environment using the following commands:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install required dependencies

```bash
pip install -r requirements.txt
```

3. Create a .env file in the root directory with the following variables

```bash
INDEX_NAME=<your-pinecone-index-name>
EMBEDDING_DIMENSION=<dimension-of-embeddings>
OPENAI_MODEL=<your-openai-model>
PINECONE_API_KEY=<your-pinecone-api-key>
OPENAI_API_KEY=<your-openai-api-key>
```


To run the script locally, use the following command:

```bash
streamlit run app.py
```


This README file provides detailed instructions for both Docker-based and local setups, including steps for environment variable configuration and application execution.
