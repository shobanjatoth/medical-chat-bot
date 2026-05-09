import os

from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Load PDF data
extracted_data = load_pdf_file(data="data/")

# Filter docs
filtered_data = filter_to_minimal_docs(
    extracted_data
)

# Split chunks
text_chunks = text_split(filtered_data)

# Embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chat-bot"

# Create index if not exists
if index_name not in pc.list_indexes().names():

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Store embeddings
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

print("Pinecone index created successfully.")