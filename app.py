import os
import openai
import hashlib
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader

INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

PROMPT = """
You are tasked to answer the question based on the provided context. The question and the context are provided at the end. Analyse the context provided to you thoroughly and provide the most relevant answer. If the question seems irrelevant to the context, return an "No relevant information found. Please try a different question." response.
"""

# Pinecone setup
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


def clear_pinecone_index():
    """Clear all vectors in the Pinecone index."""
    index.delete(delete_all=True)


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def create_embeddings(text):
    """Create embeddings for the provided text using OpenAI API."""
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    return [float(x) for x in response.data[0].embedding]


def handle_pinecone_results(results):
    """Handle the results from Pinecone query, fetch and process metadata."""
    print("Pinecone query results:", results)

    ids = [match.id for match in results.matches]
    vector_data = index.fetch(ids)

    texts = []
    for id, data in vector_data.vectors.items():
        if "metadata" in data and "text" in data["metadata"]:
            texts.append(data["metadata"]["text"])
        else:
            print(f"Warning: No text content found for id {id}")

    return " ".join(texts)


def ingest_pdf_to_pinecone(pdf_file):
    """Ingest PDF content into Pinecone."""
    text = extract_text_from_pdf(pdf_file)
    chunk_size = 1000
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = [create_embeddings(chunk) for chunk in chunks]
    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch = [
            {"id": str(j + i), "values": embeddings[j], "metadata": {"text": chunks[j]}}
            for j in range(min(batch_size, len(embeddings) - i))
        ]
        index.upsert(vectors=batch)


def query_pinecone(query):
    """Query Pinecone for similar embeddings to the provided query."""
    query_embedding = create_embeddings(query)
    response = index.query(vector=query_embedding, top_k=5)
    return response


def generate_response_from_gpt4(retrieved_text, query):
    """Generate a response from GPT based on retrieved text and the query."""
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate answers based on the provided context.",
            },
            {
                "role": "user",
                "content": f"{PROMPT}\n\n#Context:\n```\n{retrieved_text}\n```\n#Question: ```{query}```",
            },
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def file_has_changed(new_file):
    """Check if the new file is different from the previous one by comparing file hashes."""
    new_file_bytes = new_file.read()
    new_file_hash = hashlib.md5(new_file_bytes).hexdigest()
    new_file.seek(0)  # Reset file pointer after reading

    # Check if the file hash is stored in the session and compare
    if "file_hash" in st.session_state and st.session_state.file_hash == new_file_hash:
        return False
    else:
        st.session_state.file_hash = new_file_hash
        return True


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
                    answer = generate_response_from_gpt4(retrieved_text, prompt)
                else:
                    answer = "No relevant information found. Please try a different question."

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
                response_box.markdown(answer)
    except Exception as e:
        st.error("There was an internal error")
