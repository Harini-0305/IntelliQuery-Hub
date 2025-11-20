# main.py  — HackRx-optimized FastAPI app
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
import os, time, tempfile, logging, traceback
import requests
import pdfplumber

from dotenv import load_dotenv

# Vector / embeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hackrx")

# ---------- env ----------
load_dotenv()
QDRANT_URL         = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION  = os.getenv("QDRANT_COLLECTION", "hackrx_collection").strip()
TEAM_API_KEY       = os.getenv("TEAM_API_KEY", "").strip()

# ---------- app ----------
app = FastAPI(title="HackRx Retrieval API", version="1.0")

# ---------- models ----------
class QueryRequest(BaseModel):
    documents: str                  # PDF URL
    questions: List[str]            # array of questions

# ---------- clients / models ----------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# lazy constructed langchain vectorstore (reused)
_vectorstore: Optional[Qdrant] = None


# ---------- helpers ----------
def ensure_collection_exists():
    """Create collection if missing (384 dims for MiniLM)."""
    collections = qdrant_client.get_collections().collections
    if not any(c.name == QDRANT_COLLECTION for c in collections):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

def already_embedded() -> bool:
    """True if collection has any points."""
    try:
        info = qdrant_client.count(collection_name=QDRANT_COLLECTION)
        return (info.count or 0) > 0
    except Exception:
        return False

def extract_text_from_pdf(pdf_url: str, timeout: int = 25) -> str:
    r = requests.get(pdf_url, timeout=timeout)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            pt = page.extract_text()
            if pt:
                text += pt + "\n"
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return text

def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]

def build_vectorstore() -> Qdrant:
    """Return a langchain Qdrant vectorstore bound to the collection."""
    return Qdrant.from_existing_collection(
        collection_name=QDRANT_COLLECTION,
        location=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        embedding=embedding_model,
    )

def store_chunks_to_qdrant(chunks: List[Document]):
    ensure_collection_exists()
    Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=QDRANT_COLLECTION,
        location=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

def verify_bearer_or_alt(
    authorization: Optional[str] = Header(None),
    x_team_key: Optional[str] = Header(None),
):
    """
    Accept either:
      - Authorization: Bearer <TEAM_API_KEY>
      - X-Team-Key: <TEAM_API_KEY>
    """
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
    elif x_team_key:
        token = x_team_key.strip()

    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    if not TEAM_API_KEY:
        raise HTTPException(status_code=500, detail="Server token not configured")
    if token != TEAM_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")
    return True


# ---------- health ----------
@app.get("/healthz")
def healthz():
    issues = []
    if not QDRANT_URL: issues.append("missing QDRANT_URL")
    if not QDRANT_API_KEY: issues.append("missing QDRANT_API_KEY")
    if not TEAM_API_KEY: issues.append("missing TEAM_API_KEY")
    status = "ok" if not issues else "degraded"
    return {"status": status, "issues": issues}


# ---------- main endpoint (HackRx-optimized) ----------
@app.post("/hackrx/run")
def hackrx_run(req: QueryRequest, _: bool = Depends(verify_bearer_or_alt)):
    t0 = time.time()
    global _vectorstore

    try:
        # 1) Prepare index ONLY if empty (first cold start)
        if not already_embedded():
            logger.info("Collection empty; ingesting PDF once...")
            raw_text = extract_text_from_pdf(req.documents)
            docs = chunk_text(raw_text)
            store_chunks_to_qdrant(docs)
            _vectorstore = None  # rebuild below

        # 2) Get (or build) a cached vectorstore handle
        if _vectorstore is None:
            _vectorstore = build_vectorstore()

        # 3) Answer by returning the top clause EXACTLY (best for string matching)
        from concurrent.futures import ThreadPoolExecutor

        def fetch_answer(q: str) -> str:
            hits = _vectorstore.similarity_search(q, k=1)
            return hits[0].page_content.strip() if hits else "[No answer found]"

        with ThreadPoolExecutor(max_workers=max(1, len(req.questions))) as ex:
            answers = list(ex.map(fetch_answer, req.questions))

        elapsed = round(time.time() - t0, 2)
        return {
            "status": "success",
            "processing_time": f"{elapsed}s",
            "answers": answers,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("hackrx_run failed")
        raise HTTPException(status_code=500, detail=str(e))
