import os
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "hackrx_collection")

# Embedding model (384-dim)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Manual client (optional, for advanced ops)
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Make sure collection exists
def init_qdrant_collection():
    collections = client.get_collections().collections
    if not any(c.name == QDRANT_COLLECTION for c in collections):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

def store_chunks_to_qdrant(chunks: List[Document]):
    init_qdrant_collection()
    # ✅ No client= here — just location and api_key
    Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=QDRANT_COLLECTION,
        location=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

def query_similar_chunks(query: str, k: int = 5) -> List[Document]:
    vectorstore = Qdrant.from_existing_collection(
        collection_name=QDRANT_COLLECTION,
        location=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        embedding=embedding_model
    )
    return vectorstore.similarity_search(query, k=k)
# ✅ Test block
if __name__ == "__main__":
    docs = [
        Document(page_content="The grace period is 30 days."),
        Document(page_content="Maternity benefits apply after 24 months."),
        Document(page_content="Cataract surgery is covered after 2 years.")
    ]
    store_chunks_to_qdrant(docs)
    result = query_similar_chunks("What is the grace period?")
    for i, doc in enumerate(result, 1):
        print(f"{i}. {doc.page_content}")
