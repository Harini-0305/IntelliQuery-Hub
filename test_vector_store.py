# test_vector_store.py

from langchain.schema import Document
from embedder.vector_store import store_chunks_to_pinecone, query_similar_chunks

# Dummy test content
docs = [
    Document(page_content="This insurance policy includes a 30-day grace period."),
    Document(page_content="Coverage for cataract surgery is available after 2 years."),
    Document(page_content="Maternity benefits apply only after 24 months of continuous coverage.")
]

# Store the test chunks
store_chunks_to_pinecone(docs)

# Run a sample query
results = query_similar_chunks("What is the grace period?")
print("Top Results:")
for i, r in enumerate(results, 1):
    print(f"{i}. {r.page_content}")
