import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_groq import ChatGroq

load_dotenv()

class RAGSystem:
    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:") 
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Ensure collection exists
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        # Initialize Compressor
        llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
        self.compressor = LLMChainExtractor.from_llm(llm)

    def ingest_pdf(self, file_path: str):
        """Ingests a PDF file into the vector database."""
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found."

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        self.vector_store.add_documents(texts)
        return f"Successfully ingested {len(texts)} chunks from '{file_path}'."

    def retrieve(self, query: str, k: int = 5):
        """Retrieves and compresses relevant documents."""
        # Base retriever
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        # Compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, 
            base_retriever=base_retriever
        )
        
        try:
            return compression_retriever.invoke(query)
        except Exception:
             # Fallback to standard retrieval if compression fails (e.g. empty result)
            return base_retriever.invoke(query)

if __name__ == "__main__":
    # Test RAG
    # Create a dummy PDF first if needed
    rag = RAGSystem()
    print(rag.ingest_pdf("src/sample-1.pdf")) # Uncomment if you have a pdf
    print(rag.retrieve("What is the content?"))
