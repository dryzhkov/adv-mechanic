# CLAUDE.md

## Project

bike-mechanic — Agentic RAG system for motorcycle service manuals.
Combines a local PDF knowledge base (LanceDB) with real-time web search
(ADVRider, Reddit, ThumperTalk) to answer torque specs, valve clearances,
and procedural how-to questions.

## Tech Stack

- Python 3.13, managed with `uv`
- LangGraph (agentic workflow)
- LangChain + OpenAI (LLM calls)
- LanceDB (local vector store, file-based)
- sentence-transformers (local embeddings, `all-MiniLM-L6-v2`)
- pdfplumber (table-aware PDF extraction)
- Tavily (web search API)
- MCP (tool servers for RAG and web search)
- Typer (CLI)

## Development

### Setup

```sh
uv sync
cp .env.example .env  # then fill in API keys
```

### Ingest manuals

```sh
uv run bike-mechanic ingest --all       # all PDFs in data/manuals/
uv run bike-mechanic ingest path/to.pdf  # single PDF
```

### Chat

```sh
uv run bike-mechanic chat
uv run bike-mechanic chat --bike "KTM 890 Adventure R"
uv run bike-mechanic ask "cam chain tensioner torque" --bike "KTM 890 Adventure R"
```

### MCP servers

```sh
uv run python servers/rag_server.py   # RAG tools
uv run python servers/web_server.py   # Web search tools
```

### Test

```sh
uv run pytest
```

### Lint

```sh
uv run ruff check .
```

## Project Structure

```
bike-mechanic/
├── data/
│   ├── manuals/              # source PDFs (gitignored)
│   └── vectorstore/          # LanceDB files (gitignored)
├── servers/
│   ├── rag_server.py         # MCP server: manual search tools
│   └── web_server.py         # MCP server: web search tools
├── src/bike_mechanic/
│   ├── main.py               # CLI entry point (typer)
│   ├── graph.py              # LangGraph workflow definition
│   ├── state.py              # Graph state schema
│   ├── config.py             # Settings from env vars
│   ├── search.py             # Shared vector store search logic
│   ├── nodes/
│   │   ├── router.py         # Query classification
│   │   ├── retrieve.py       # RAG retrieval from LanceDB
│   │   ├── grade.py          # Relevance/completeness grading
│   │   ├── web_search.py     # Web search via Tavily
│   │   ├── conflict.py       # Manual vs web conflict resolution
│   │   └── generate.py       # Final answer with citations
│   └── ingestion/
│       ├── pipeline.py       # Ingestion orchestrator
│       ├── pdf_parser.py     # pdfplumber extraction + table detection
│       └── chunker.py        # Table-aware chunking
├── pyproject.toml
├── .env                      # API keys (gitignored)
└── CLAUDE.md
```

## Code Style

- Follow existing patterns and conventions in the codebase
- Keep functions small and focused
- Use meaningful, descriptive names
- Type hints on function signatures

## Key Design Decisions

- Tables from PDFs are kept as atomic chunks (never split) to preserve
  torque spec row-column relationships
- LanceDB chosen for zero-server-process deployment (files on disk)
- LangGraph nodes call search functions directly (not via MCP) for speed;
  MCP servers expose the same logic for external AI clients
- Safety-critical components trigger automatic disclaimers
