import os, sys
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "vectordb", "chroma")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_PROVIDER = os.getenv("LLM_PROVIDER","ollama").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","llama3.1:8b")

def coll():
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=False))
    return client.get_or_create_collection("docs")

def call_llm(prompt: str) -> str:
    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        return OpenAI().chat.completions.create(
            model=OPENAI_MODEL, temperature=0.2,
            messages=[{"role":"user","content":prompt}]
        ).choices[0].message.content
    else:
        import ollama
        return ollama.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":prompt}])["message"]["content"]

def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.ask_chroma "your question" [k]'); sys.exit(1)
    question = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv)>2 else 5
    c = coll()
    if c.count()==0:
        print("Index empty. Run: python -m src.ingest_chroma"); sys.exit(1)

    embedder = SentenceTransformer(EMBED_MODEL)
    # IMPORTANT: return a numpy array, then convert to list-of-lists
    q_emb = embedder.encode([question], convert_to_numpy=True).tolist()

    res = c.query(query_embeddings=q_emb, n_results=k, include=["documents","metadatas"])
    docs = res["documents"][0]; metas = res["metadatas"][0]

    ctx = "\n\n".join(f"- {d}" for d in docs)
    prompt = f"""Use ONLY the context to answer. If not in context, say "I don't know from the provided context."

Context:
{ctx}

Question: {question}

Answer:"""

    ans = call_llm(prompt)
    print("\n=== ANSWER ===\n", ans.strip())
    print("\n=== SOURCES ===")
    for m in metas:
        print(f"{m.get('source','?')} (page {m.get('page','?')})")

if __name__ == "__main__":
    main()
