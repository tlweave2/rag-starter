# rag-tutorial-v2

## Quick demo
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.ingest_chroma
export LLM_PROVIDER=ollama     # or openai
python -m src.ask_chroma "How do I build a hotel in Monopoly?"
```
