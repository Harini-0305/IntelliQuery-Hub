import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "hackrx_collection"
EMBED_DIM = 384  # for sentence-transformers/all-MiniLM-L6-v2

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
)

print(f"[OK] Recreated collection '{COLLECTION_NAME}' with dim={EMBED_DIM}")
