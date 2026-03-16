# adv-mechanic

Agentic RAG system for adventure motorcycle service manuals. Combines a local PDF knowledge base (LanceDB) with real-time web search (ADVRider, Reddit, ThumperTalk) to answer torque specs, valve clearances, and procedural how-to questions.

Built with LangGraph, LangChain, and MCP.

## How it works

```
User Question
     │
     ▼
┌─────────┐     ┌──────────┐     ┌─────────┐
│  Router  │────▶│ Retrieve │────▶│  Grade  │
│(classify)│     │(LanceDB) │     │(LLM)    │
└─────────┘     └──────────┘     └────┬────┘
                                      │
                        ┌─────────────┼─────────────┐
                        ▼ sufficient   ▼ partial/     │
                   ┌─────────┐    insufficient   │
                   │Generate │    ┌───────────┐  │
                   │(answer) │    │Web Search │  │
                   └─────────┘    │(Tavily)   │  │
                        ▲         └─────┬─────┘  │
                        │               ▼        │
                        │      ┌──────────────┐  │
                        │      │  Conflict    │  │
                        └──────│  Resolution  │──┘
                               └──────────────┘
```

The system:
1. **Routes** the query (lookup vs. procedural vs. general)
2. **Retrieves** relevant chunks from the local vector store (hybrid vector + full-text search)
3. **Grades** retrieval quality — for lookup queries, checks that actual numeric specs are present
4. **Searches the web** if manual data is insufficient (ADVRider, Reddit, ThumperTalk)
5. **Resolves conflicts** between manual and community data
6. **Generates** a cited answer with safety disclaimers for critical fasteners
7. **Scores confidence** (0–100%) based on retrieval quality, source verification, and conflict status

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **OpenAI API key** — for LLM calls (GPT-4o-mini by default)
- **Tavily API key** *(optional)* — for web search validation ([free tier](https://tavily.com))

## Setup

```sh
# Clone the repo
git clone https://github.com/dryzhkov/adv-mechanic.git
cd adv-mechanic

# Install dependencies
uv sync

# Configure API keys
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...  (optional, enables web search)
```

## Add service manuals

Place PDF service manuals in `data/manuals/`:

```sh
mkdir -p data/manuals
cp ~/Downloads/2022_KTM_890_Shop_Manual.pdf data/manuals/
```

The system auto-detects bike model and year from the filename. Supported naming patterns:
- `*890*` → KTM 890 Adventure R
- `*te*300*` → Husqvarna TE 300
- `*te*250*` → Husqvarna TE 250

## Ingest manuals

```sh
# Ingest all PDFs in data/manuals/
uv run adv-mechanic ingest --all

# Or ingest a single file
uv run adv-mechanic ingest data/manuals/2022_KTM_890_Shop_Manual.pdf
```

Ingestion is re-runnable — running it again on the same manual replaces the old chunks.

The pipeline:
1. Extracts text and tables from each PDF page (with watermark filtering)
2. Keeps tables as atomic chunks (torque spec tables are never split)
3. Splits text into overlapping chunks (~800 chars)
4. Embeds with `all-MiniLM-L6-v2` (runs locally, no API key needed)
5. Stores in LanceDB (file-based, no server process)
6. Builds a full-text search index for exact spec lookups

## Usage

### Interactive chat

```sh
uv run adv-mechanic chat
uv run adv-mechanic chat --bike "KTM 890 Adventure R"
```

### Single question

```sh
uv run adv-mechanic ask "front wheel spindle torque" --bike "KTM 890 Adventure R"
uv run adv-mechanic ask "How do I adjust the valves?" --bike "Husqvarna TE 300"
```

Each answer includes a confidence score:

```
[Confidence: HIGH 100%]

The front wheel spindle torque is 45 Nm (33.2 lb-ft) for the M25x1.5 screw,
with the thread greased (per service manual p.151).
```

The score is computed from retrieval grade, source confidence, spec verification against source text, and conflict status. `HIGH` (75–100%), `MEDIUM` (40–74%), `LOW` (0–39%).

### MCP servers

Expose the same search tools to external AI clients (e.g., Claude Desktop):

```sh
uv run python servers/rag_server.py   # Manual search tools
uv run python servers/web_server.py   # Web search tools
```

## Configuration

All settings are via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for LLM calls |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model (local) |
| `TAVILY_API_KEY` | *(optional)* | Enables web search validation |
| `SIMILARITY_THRESHOLD` | `1.5` | Max L2 distance for vector search results |

## Project structure

```
adv-mechanic/
├── data/
│   ├── manuals/              # Source PDFs (gitignored)
│   └── vectorstore/          # LanceDB files (gitignored)
├── servers/
│   ├── rag_server.py         # MCP server: manual search tools
│   └── web_server.py         # MCP server: web search tools
├── src/adv_mechanic/
│   ├── main.py               # CLI entry point (Typer)
│   ├── graph.py              # LangGraph workflow
│   ├── state.py              # Graph state schema
│   ├── config.py             # Settings from env vars
│   ├── search.py             # Vector store search (vector + FTS)
│   ├── nodes/
│   │   ├── router.py         # Query classification
│   │   ├── retrieve.py       # RAG retrieval from LanceDB
│   │   ├── grade.py          # Relevance/completeness grading
│   │   ├── web_search.py     # Web search via Tavily
│   │   ├── conflict.py       # Manual vs web conflict resolution
│   │   └── generate.py       # Final answer with citations
│   └── ingestion/
│       ├── pipeline.py       # Ingestion orchestrator
│       ├── pdf_parser.py     # pdfplumber extraction + watermark filtering
│       └── chunker.py        # Table-aware chunking
├── pyproject.toml
├── .env.example              # Template for API keys
└── CLAUDE.md
```

## Development

```sh
# Run tests
uv run pytest

# Lint
uv run ruff check .
```

## Key design decisions

- **Table-aware chunking** — Tables from PDFs are kept as atomic chunks to preserve torque spec row-column relationships
- **Watermark filtering** — PDF extraction filters out DRM watermark layers (rotated text, margin annotations) that otherwise garble the extracted content
- **Hybrid search** — Combines vector similarity with full-text search (tantivy) for lookup queries, so exact values like "45Nm" aren't missed by embedding distance alone
- **Anti-hallucination** — The generation prompt requires every numeric spec to appear verbatim in source text; a post-generation check flags any unverified values
- **Confidence scoring** — Each answer gets a 0–100% score derived from retrieval grade, retrieval confidence, source type (manual vs. web-only), spec verification, and conflict status
- **Safety disclaimers** — Queries about safety-critical components (brakes, axles, steering) automatically include torque wrench / verification warnings
- **LanceDB** — Zero-server-process vector store (files on disk), easy to deploy
- **Direct function calls** — LangGraph nodes call search functions directly for speed; MCP servers expose the same logic for external AI clients
