import os, hashlib
from typing import List, Tuple, Dict
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE, "data")
DB_DIR = os.path.join(BASE, "vectordb", "chroma")
EMBED_MODEL = "all-MiniLM-L6-v2"

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_chunks() -> List[Tuple[str, Dict]]:
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    items, counts = [], {}
    for d in chunks:
        src = d.metadata.get("source", "unknown")
        page = int(d.metadata.get("page", 0))
        key = (src, page)
        idx = counts.get(key, 0)
        counts[key] = idx + 1
        text = (d.page_content or "").strip()
        cid  = f"{src}#page-{page}-chunk-{idx}"
        items.append((text, {"id": cid, "source": src, "page": page, "chunk": idx, "sha1": sha1(text)}))
    return items

def get_collection():
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=False))
    return client.get_or_create_collection("docs")

def main():
    items = load_chunks()
    print(f"prepared {len(items)} chunks")

    coll = get_collection()
    # just add everything; embeddings are deterministic anyway
    add_ids, add_docs, add_meta = [], [], []
    for text, meta in items:
        add_ids.append(meta["id"]); add_docs.append(text); add_meta.append(meta)

    embedder = SentenceTransformer(EMBED_MODEL)

    # IMPORTANT: force numpy arrays -> plain lists for Chroma
    embs = embedder.encode(add_docs, convert_to_numpy=True)  # numpy array (N, D)
    embs = embs.tolist()  # -> List[List[float]]

    # write
    coll.add(ids=add_ids, documents=add_docs, metadatas=add_meta, embeddings=embs)
    print(f"added {len(add_ids)} chunks")
    print(f"done. collection size: {coll.count()}")

if __name__ == "__main__":
    main()
