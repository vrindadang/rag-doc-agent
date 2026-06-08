# RAG Doc Agent — Multimodal RAG + LangGraph Documentation Agent

A two-in-one project: a **multimodal RAG chatbot** that answers questions from your PDFs (including tables and charts), and a **LangGraph agentic loop** that automatically rewrites documentation on every GitHub push.

> Built to explore production RAG patterns, PDF vision extraction, and multi-agent workflows with Modifier/Reviewer agents.

---

## What It Does

### Tab 1 — RAG Chat
- Upload any PDF and ask questions about it — answers are grounded in the document, no hallucination
- Multimodal ingestion: extracts native text **+** structured tables (PyMuPDF) **+** visual chart/image descriptions (Azure OpenAI vision)
- Answers stream token-by-token with source citations showing the page and filename
- Vector similarity search via **Supabase pgvector**

### Tab 2 — Auto-Updated Documentation
- Every push to `app/` triggers a GitHub Actions workflow
- A **Modifier agent** reads the code diff and rewrites the docs
- A **Reviewer agent** checks accuracy and requests revisions if needed
- Loops up to 3 times until approved, then stores the final version in Supabase
- All agent iterations and reasoning are logged and viewable in the UI

---

## Tech Stack

| Component | Technology |
|---|---|
| RAG Backend | FastAPI + Supabase pgvector |
| LLM / Embeddings | Azure OpenAI (GPT-4o + text-embedding-3-small) |
| PDF Parsing | PyMuPDF (text + tables + vision) |
| Agentic Workflow | LangGraph (StateGraph, Modifier/Reviewer loop) |
| UI | Streamlit (two-tab layout) |
| CI/CD | GitHub Actions |
| Storage | Supabase (document chunks, doc versions, agent logs) |

---

## Project Structure

```
rag-doc-agent/
├── app/
│   ├── main.py           # FastAPI app — /ingest, /query, /documents endpoints
│   ├── ingestor.py       # PDF extraction: text + tables + vision → embeddings → Supabase
│   ├── retriever.py      # Embed question → pgvector similarity search
│   ├── generator.py      # Prompt builder + Azure OpenAI streaming response
│   ├── prompt_builder.py # RAG prompt construction with citations
│   └── config.py         # Environment variable loader
├── agents/
│   ├── doc_updater.py    # LangGraph Modifier/Reviewer agent loop
│   └── supabase_store.py # Supabase read/write: docs, versions, agent logs
├── ui/
│   └── streamlit_app.py  # Two-tab Streamlit interface
├── .github/
│   └── workflows/
│       └── update-docs.yml  # GitHub Actions: triggers doc agent on push
├── sample_docs/          # Auto-ingested sample PDFs
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### Prerequisites
- Python 3.11+
- A [Supabase](https://supabase.com) project with the pgvector extension enabled
- An Azure OpenAI resource with a chat deployment (GPT-4o recommended) and an embedding deployment

### 1. Clone and install

```bash
git clone https://github.com/vrindadang/rag-doc-agent.git
cd rag-doc-agent
python -m venv venv
# Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
CHAT_DEPLOYMENT=gpt-4o
AZURE_CHAT_DEPLOYMENT=gpt-4o
EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Supabase
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_ANON_KEY=<your-anon-key>
SUPABASE_SERVICE_KEY=<your-service-key>
```

### 3. Set up Supabase tables

Run the following SQL in your Supabase SQL editor:

```sql
-- Enable pgvector
create extension if not exists vector;

-- Document chunks table
create table document_chunks (
  id bigserial primary key,
  filename text,
  page_num int,
  chunk_index int,
  content text,
  embedding vector(1536)
);

-- Ingested files tracker
create table ingested_files (
  filename text primary key,
  chunk_count int,
  created_at timestamptz default now()
);

-- Documentation versions
create table doc_versions (
  id bigserial primary key,
  version int unique,
  storage_path text,
  commit_sha text,
  trigger_source text,
  created_at timestamptz default now()
);

-- Agent step logs
create table agent_logs (
  id bigserial primary key,
  version_id bigint references doc_versions(id),
  agent_name text,
  iteration int,
  message text,
  created_at timestamptz default now()
);

-- Similarity search function
create or replace function match_chunks(
  query_embedding vector(1536),
  match_count int default 3
)
returns table (
  filename text,
  page_num int,
  content text,
  similarity float
)
language plpgsql as $$
begin
  return query
  select
    document_chunks.filename,
    document_chunks.page_num,
    document_chunks.content,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from document_chunks
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

### 4. Run the Streamlit app

```bash
streamlit run ui/streamlit_app.py
```

### 5. (Optional) Run the FastAPI backend separately

```bash
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`

---

## GitHub Actions — Auto Docs on Push

When code in `app/` changes, the workflow in `.github/workflows/update-docs.yml` triggers automatically:

1. Checks out the repo
2. Runs `agents/doc_updater.py` with the Git diff as input
3. The Modifier agent rewrites the docs; the Reviewer agent checks accuracy
4. The final approved version is stored in Supabase and visible in the UI

To enable: add your environment variables as GitHub Actions secrets (`AZURE_OPENAI_API_KEY`, `SUPABASE_URL`, etc.).

---

## How Multimodal Ingestion Works

For each PDF page, the ingestor runs three extraction passes in sequence:

1. **Native text** via `page.get_text()` — fast and accurate for digital PDFs
2. **Structured tables** via `page.find_tables()` — extracts table rows as pipe-separated text
3. **Vision summary** via Azure OpenAI GPT-4o — describes charts, graphs, and image-heavy pages

All three are combined into a single searchable text chunk before embedding.

---

## Security Note

Never commit your `.env` file. The `.gitignore` already excludes it. For GitHub Actions, store secrets in your repository's **Settings → Secrets and variables → Actions**.

---
