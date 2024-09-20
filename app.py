import os
import openai
from openai import AssistantEventHandler
import hashlib
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

# Configuration
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


PROMPT = """
You are tasked to answer the question based on the provided context. The question and the context are provided at the end. Analyze the context thoroughly and provide the most relevant answer. Additionally, indicate the page number from which the information was sourced.

If the question seems irrelevant to the context or chat history, return the following response:
"No relevant information found. Please try a different question."
"""

# OpenAI and Pinecone setup

try:
    openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # Create the index if it doesn't exist
    if INDEX_NAME not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # Connect to the index
    index = pinecone.Index(INDEX_NAME)

except Exception as e:
    print("Error initializing OpenAI or Pinecone:", e)
    st.error("Failed to initialize services. Please check your configuration.")


class EventHandler(AssistantEventHandler):
    def __init__(self, response_box):
        super().__init__()  # Call the parent constructor
        self.response_box = response_box
        self.result_output = []

    def on_text_created(self, text):
        print("\nassistant > ", end="", flush=True)

    def on_text_delta(self, delta, snapshot):
        self.result_output.append(delta.value)
        result = "".join(self.result_output).strip()
        self.response_box.markdown(result, unsafe_allow_html=True)
        print(delta.value, end="", flush=True)


def clear_pinecone_index():
    """Clear all vectors in the Pinecone index."""
    try:
        index_stats = index.describe_index_stats()
        if index_stats["total_vector_count"] > 0:
            index.delete(delete_all=True)
            print("Index cleared.")
        else:
            print("Index is already empty. No need to clear.")
    except Exception as e:
        print("Error clearing Pinecone index:", e)
        st.error("Failed to clear the Pinecone index.")


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print("Error extracting text from PDF:", e)
        st.error("Failed to extract text from the PDF.")


def create_embeddings(text: str) -> list:
    """Create embeddings for the provided text using OpenAI API."""
    try:
        response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
        return [float(x) for x in response.data[0].embedding]
    except Exception as e:
        print("Error creating embeddings:", e)
        st.error("Failed to create embeddings.")


def handle_pinecone_results(results):
    """Handle the results from Pinecone query, fetch and process metadata."""
    try:
        ids = [match.id for match in results.matches]
        vector_data = index.fetch(ids)

        texts = []
        for id, data in vector_data.vectors.items():
            if "metadata" in data and "text" in data["metadata"]:
                texts.append(data["metadata"]["text"])
            else:
                print(f"Warning: No text content found for id {id}")

        return " ".join(texts)
    except Exception as e:
        print("Error handling Pinecone results:", e)
        st.error("Failed to handle results from Pinecone.")


def ingest_pdf_to_pinecone(pdf_file):
    """Ingest PDF content into Pinecone."""
    try:
        text = extract_text_from_pdf(pdf_file)
        chunk_size = 1000
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = [create_embeddings(chunk) for chunk in chunks]
        batch_size = 100
        for i in range(0, len(embeddings), batch_size):
            batch = [
                {
                    "id": str(j + i),
                    "values": embeddings[j],
                    "metadata": {"text": chunks[j]},
                }
                for j in range(min(batch_size, len(embeddings) - i))
            ]
            index.upsert(vectors=batch)
    except Exception as e:
        print("Error ingesting PDF to Pinecone:", e)
        st.error("Failed to ingest PDF into Pinecone.")


def query_pinecone(query: str):
    """Query Pinecone for similar embeddings to the provided query."""
    try:
        query_embedding = create_embeddings(query)
        response = index.query(vector=query_embedding, top_k=5)
        return response
    except Exception as e:
        print("Error querying Pinecone:", e)
        st.error("Failed to query Pinecone.")


def file_has_changed(new_file) -> bool:
    """Check if the new file is different from the previous one by comparing file hashes."""
    try:
        new_file_bytes = new_file.read()
        new_file_hash = hashlib.md5(new_file_bytes).hexdigest()
        new_file.seek(0)  # Reset file pointer after reading

        # Check if the file hash is stored in the session and compare
        if (
            "file_hash" in st.session_state
            and st.session_state.file_hash == new_file_hash
        ):
            return False
        else:
            st.session_state.file_hash = new_file_hash
            return True
    except Exception as e:
        print("Error checking if file has changed:", e)
        st.error("Failed to check file hash.")


# Streamlit app
if __name__ == "__main__":
    try:
        st.markdown(
            """<style>.block-container{max-width: 66rem !important;}</style>""",
            unsafe_allow_html=True,
        )
        st.title("RAG Chatbot")

        # Initialize session state for chat history and file upload state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "file_uploaded" not in st.session_state:
            st.session_state.file_uploaded = False

        if "assistant" not in st.session_state:
            st.session_state.assistant = openai_client.beta.assistants.create(
                name="RAG Chatbot", instructions=PROMPT, model=OPENAI_MODEL
            )

        if "thread" not in st.session_state:
            st.session_state.thread = openai_client.beta.threads.create()

        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded_file is not None and file_has_changed(uploaded_file):
            # Clear the index before ingesting new data
            st.session_state.chat_history = []
            clear_pinecone_index()
            st.write("Processing your file...")
            ingest_pdf_to_pinecone(uploaded_file)
            st.write("File ingested successfully!")
            st.session_state.file_uploaded = True
        elif uploaded_file is not None:
            st.session_state.file_uploaded = True
        else:
            st.session_state.chat_history = []
            st.session_state.file_uploaded = False

        # Display chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle input text
        prompt_disabled = (
            not st.session_state.file_uploaded
        )  # Disable input if no file is uploaded
        prompt = st.chat_input("Type your question:", disabled=prompt_disabled)

        # Handle input text
        if prompt and not prompt_disabled:
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_box = st.empty()
                response_box.markdown("Processing...")

                results = query_pinecone(prompt)
                retrieved_text = handle_pinecone_results(results)

                if retrieved_text:
                    openai_client.beta.threads.messages.create(
                        thread_id=st.session_state.thread.id,
                        role="user",
                        content=f"{PROMPT}\n\n#Context:\n```\n{retrieved_text}\n```\n#Question: ```{prompt}```",
                    )

                    event_handler = EventHandler(response_box)

                    with openai_client.beta.threads.runs.stream(
                        thread_id=st.session_state.thread.id,
                        assistant_id=st.session_state.assistant.id,  # Replace with your assistant ID
                        event_handler=event_handler,
                    ) as stream:
                        stream.until_done()

                    assistant_response = "".join(event_handler.result_output)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": assistant_response}
                    )

                    response_box.markdown(assistant_response)
                else:
                    answer = "No relevant information found. Please try a different question."
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    response_box.markdown(answer)

    except Exception as e:
        print("Internal Error: ", e)
        st.error("There was an internal error")
