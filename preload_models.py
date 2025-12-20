from langchain_huggingface import HuggingFaceEmbeddings
import os

print("Starting model download for caching...")
try:
    # Use the same parameters as in src/rag.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    # Perform a dummy encoding to trigger download
    embeddings.embed_query("hello world")
    print("Model successfully downloaded and cached.")
except Exception as e:
    print(f"Error downloading model: {e}")
    exit(1)
