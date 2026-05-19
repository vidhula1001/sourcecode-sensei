# SourceCode-Sensei 

A local, privacy-first AI coding assistant that lets you chat with your own codebase. Built with Streamlit, LangChain, FAISS, and Ollama — no cloud APIs, no data leaving your machine.

---

## What It Does

Upload your source files and ask natural language questions about them. SourceCode-Sensei retrieves the most relevant code chunks and uses a local LLM to explain, debug, or discuss your code.

- **"How does the login logic work?"**
- **"What happens when a task is deleted?"**
- **"Show me how data is stored in tracker_logic.py"**

---

## Features

-  **Hybrid Retrieval** — combines BM25 keyword search and FAISS semantic search for more accurate results
-  **Line-aware chunking** — responses show exactly which lines were referenced (e.g. `student_manager.py — Lines 12–45`)
-  **Relevance score filtering** — weak matches are automatically dropped so the LLM isn't confused by irrelevant context
-  **Session management** — start new chats, auto-named and archived by topic
-  **File management** — upload, delete, or clear all indexed files from the sidebar
-  **Context window gauge** — live progress bar showing how much of the model's context window your codebase is using
-  **Multi-language support** — Python, C, C++, JavaScript, TypeScript, Java, Go
-  **Fully local** — runs on your machine using Ollama, no internet required after setup

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | Ollama (`qwen2.5-coder:1.5b`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| Keyword Search | BM25 (LangChain) |
| Chunking | LangChain RecursiveCharacterTextSplitter |

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 2. Pull the model

```bash
ollama pull qwen2.5-coder:1.5b
```

### 3. Clone and install dependencies

```bash
git clone https://github.com/vidhula1001/codesensei.git
cd codesensei
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 4. Add your code files

Place your source files inside the `my_code/` folder:

```
codesensei/
├── my_code/
│   ├── your_file.py
│   ├── another_file.js
│   └── ...
```

### 5. Index your codebase

```bash
py ingest.py
```

### 6. Run the app

```bash
streamlit run app.py
```

---

## Project Structure

```
codesensei/
├── my_code/              # Your source files go here
├── faiss_code_index/     # Auto-generated vector database
│   ├── index.faiss
│   └── index.pkl
├── venv/                 # Virtual environment
├── app.py                # Main Streamlit app
├── ingest.py             # Indexing pipeline
├── style.css             # Custom styling
└── README.md
```

---

## How It Works

1. **Ingestion** (`ingest.py`) — loads each source file, splits it into language-aware chunks using LangChain's built-in splitters, records start/end line numbers per chunk, and saves everything to a local FAISS vector database.

2. **Retrieval** (`app.py`) — when you ask a question, a hybrid retriever runs both BM25 (keyword match) and FAISS (semantic similarity) in parallel, deduplicates results, and filters out low-relevance chunks using a score threshold.

3. **Generation** — the retrieved chunks are passed as context to the local Qwen model via Ollama, along with your question and conversation history. The model responds with code-aware explanations.

---

## Configuration

| Setting | Location | Default |
|---|---|---|
| LLM model | `app.py` → `load_system()` | `qwen2.5-coder:1.5b` |
| Embedding model | `app.py` + `ingest.py` | `all-MiniLM-L6-v2` |
| Retrieval k (chunks) | `app.py` → `load_system()` | `4` |
| Relevance threshold | `app.py` → `RELEVANCE_THRESHOLD` | `1.2` |
| Chunk size | `ingest.py` | `1000` chars |
| Chunk overlap | `ingest.py` | `100` chars |
| Max context tokens | `app.py` → `MAX_CONTEXT_TOKENS` | `32768` |

---

## Supported File Types

`.py` `.c` `.h` `.cpp` `.js` `.ts` `.java` `.go`

---

## Notes

- The context window gauge is an approximation (1 token ≈ 4 characters). It's useful as a signal, not an exact measurement.
- If the model says it can't find something in the codebase, it genuinely isn't in the indexed files — the prompt instructs it not to hallucinate connections.
- Re-run `py ingest.py` any time you add, remove, or modify files in `my_code/`.

---
