import os
from dotenv import load_dotenv
import httpx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
# from langchain_community.retrievers import ContextualCompressionRetriever
# from langchain_community.retrievers. import LLMChainExtractor
# from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

load_dotenv()

class RAGSystem:
    def __init__(self, collection_name: str = "test_rag_collection"):
        self.collection_name = collection_name
        self.uploaded_pdfs = {}  # Track uploaded PDFs: {filename: {chunks: int, doc_ids: []}}
        try:
            self.client = QdrantClient(":memory:") 
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text",base_url="http://localhost:11434")
            
            # Ensure collection exists
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )

            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            
            # Initialize Compressor
            # llm was here, removed to avoid early API key requirement since compressor is commented out
            # If compressor is needed, pass api_key to __init__

            # self.compressor = LLMChainExtractor.from_llm(llm)
            self.initialized = True
        except Exception as e:
            print(f"Warning: RAG system initialization failed: {e}")
            self.initialized = False

    def ingest_pdf(self, file_path: str, filename: str):
        """Ingests a PDF file into the vector database."""
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found."
        
        # Check if already uploaded
        if filename in self.uploaded_pdfs:
            return f"PDF '{filename}' is already uploaded."

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Add metadata to track which document each chunk belongs to
        for doc in texts:
            doc.metadata['source_file'] = filename
        
        # Add documents and get IDs
        doc_ids = self.vector_store.add_documents(texts)
        
        # Track uploaded PDF
        self.uploaded_pdfs[filename] = {
            'chunks': len(texts),
            'doc_ids': doc_ids if doc_ids else []
        }
        
        return f"Successfully ingested {len(texts)} chunks from '{filename}'."

    def delete_pdf(self, filename: str):
        """Deletes a PDF and its embeddings from the vector store."""
        if filename not in self.uploaded_pdfs:
            return f"Error: PDF '{filename}' not found."
        
        try:
            # Get all points and filter by source_file metadata
            # Since we're using in-memory Qdrant, we'll delete by filtering
            pdf_info = self.uploaded_pdfs[filename]
            
            # Delete from Qdrant by IDs if available
            if pdf_info.get('doc_ids'):
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=pdf_info['doc_ids']
                )
            
            # Remove from tracking
            del self.uploaded_pdfs[filename]
            
            return f"Successfully deleted '{filename}' and its {pdf_info['chunks']} chunks."
        except Exception as e:
            return f"Error deleting '{filename}': {e}"
    
    def get_uploaded_pdfs(self):
        """Returns list of uploaded PDF filenames."""
        return list(self.uploaded_pdfs.keys())
    
    def retrieve(self, query: str, k: int = 5):
        """Retrieves and compresses relevant documents."""
        if not self.initialized:
            print("RAG system not properly initialized")
            return []
        
        try:
            # Base retriever
            base_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            
            # Compression retriever
            # compression_retriever = ContextualCompressionRetriever(
            #     base_compressor=self.compressor, 
            #     base_retriever=base_retriever
            # )
            
            results = base_retriever.invoke(query)
            print(f"RAG retrieved {len(results)} documents")
            return results
        except Exception as e:
            print(f"Error in RAG retrieval: {e}")
            return []

if __name__ == "__main__":
    # Test RAG
    # Create a dummy PDF first if needed
    rag = RAGSystem()
    print(rag.ingest_pdf("src/sample-1.pdf")) # Uncomment if you have a pdf
    print(rag.retrieve("What is the content?"))
