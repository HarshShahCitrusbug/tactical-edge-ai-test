# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the .env file from the root directory to the container
COPY .env /app/.env

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the container loads the environment variables from .env file
RUN pip install python-dotenv

# Set environment variables (optional, in case you want defaults here or overwrite)
# ENV INDEX_NAME=rag-index \
#     EMBEDDING_DIMENSION=1536 \
#     OPENAI_MODEL=gpt-4o \
#     PINECONE_API_KEY=<PINECONE_API_KEY> \
#     OPENAI_API_KEY=<OPENAI_API_KEY>

# Run Streamlit when the container launches
CMD ["streamlit", "run", "app.py"]
