"""Shared search/retrieval logic for the manual vector store."""

import logging
import os
import re
from dataclasses import dataclass

# Suppress noisy model-loading warnings before importing the libraries
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
os.environ.setdefault("ST_LOAD_REPORT", "0")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import lancedb  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from adv_mechanic.config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD, VECTORSTORE_DIR

# Module-level caches
_db = None
_embedding_model = None


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    text: str
    page_number: int
    content_type: str
    section: str
    manual_title: str
    bike_model: str
    model_year: int | None
    score: float


def _get_db():
    global _db
    if _db is None:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        _db = lancedb.connect(str(VECTORSTORE_DIR))
    return _db


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # The safetensors Rust layer writes LOAD REPORT directly to the OS
        # file descriptors (not Python sys.stdout/stderr), so we must
        # redirect at the fd level to suppress it.
        import sys

        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            sys.stdout.flush()
            sys.stderr.flush()
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(devnull)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
    return _embedding_model


def _row_to_result(r: dict) -> SearchResult:
    return SearchResult(
        text=r["text"],
        page_number=r["page_number"],
        content_type=r["content_type"],
        section=r.get("section", ""),
        manual_title=r.get("manual_title", ""),
        bike_model=r.get("bike_model", ""),
        model_year=r.get("model_year"),
        score=r.get("_distance", 0.0),
    )


def _get_known_models() -> list[str]:
    """Return the distinct bike_model values stored in LanceDB."""
    db = _get_db()
    if "manuals" not in db.table_names():
        return []
    table = db.open_table("manuals")
    arrow = table.to_arrow()
    col = arrow.column("bike_model").to_pylist()
    return list({v for v in col if v})


def resolve_bike_model(user_input: str) -> str:
    """Resolve a user's shorthand bike name to the canonical model in the store.

    Matching strategy (first match wins):
      1. Exact match (case-insensitive)
      2. User input is a substring of a stored model  ("890" -> "KTM 890 Adventure R")
      3. Stored model is a substring of user input     ("KTM 890 Adventure R" in "my ktm 890 adventure r 2022")
      4. All tokens in user input appear in a stored model ("te 300" -> "Husqvarna TE 300")

    Returns the canonical model string, or the original input if no match is found.
    """
    if not user_input:
        return ""

    known = _get_known_models()
    if not known:
        return user_input

    lower = user_input.strip().lower()

    # 1. Exact match
    for model in known:
        if model.lower() == lower:
            return model

    # 2. User input is a substring of a stored model
    for model in known:
        if lower in model.lower():
            return model

    # 3. Stored model is a substring of user input
    for model in known:
        if model.lower() in lower:
            return model

    # 4. All tokens in user input appear in a stored model
    tokens = lower.split()
    for model in known:
        model_lower = model.lower()
        if all(t in model_lower for t in tokens):
            return model

    # 5. Normalize both sides (strip spaces/punctuation) and check substrings
    normalized_input = re.sub(r"[\s\-_]+", "", lower)
    for model in known:
        normalized_model = re.sub(r"[\s\-_]+", "", model.lower())
        if normalized_input in normalized_model or normalized_model in normalized_input:
            return model

    # No match — return as-is and let downstream handle it
    return user_input


def search_manuals(
    query: str,
    bike_model: str = "",
    content_type: str = "",
    top_k: int = 10,
    distance_threshold: float | None = None,
) -> list[SearchResult]:
    """Semantic search over ingested manuals.

    Results with _distance > distance_threshold are filtered out.
    """
    if distance_threshold is None:
        distance_threshold = SIMILARITY_THRESHOLD

    db = _get_db()

    if "manuals" not in db.table_names():
        return []

    table = db.open_table("manuals")
    model = _get_embedding_model()
    query_embedding = model.encode(query)

    search = table.search(query_embedding).limit(top_k)

    # Apply metadata filters
    filters = []
    if bike_model:
        filters.append(f'bike_model = "{bike_model.replace(chr(34), "")}"')
    if content_type:
        filters.append(f'content_type = "{content_type.replace(chr(34), "")}"')
    if filters:
        try:
            search = search.where(" AND ".join(filters))
        except Exception:
            pass

    rows = search.to_list()

    # Filter by distance threshold
    return [
        _row_to_result(r)
        for r in rows
        if r.get("_distance", 999) <= distance_threshold
    ]


def search_manuals_fts(
    query: str,
    bike_model: str = "",
    top_k: int = 10,
) -> list[SearchResult]:
    """Full-text search over ingested manuals.

    Useful for finding exact spec values (e.g., "45Nm") that
    embedding similarity may miss.
    """
    db = _get_db()

    if "manuals" not in db.table_names():
        return []

    table = db.open_table("manuals")

    try:
        results = table.search(query, query_type="fts").limit(top_k)
        if bike_model:
            results = results.where(f'bike_model = "{bike_model.replace(chr(34), "")}"')
        rows = results.to_list()
    except Exception:
        # FTS index may not exist yet
        return []

    return [_row_to_result(r) for r in rows]


def search_manuals_hybrid(
    query: str,
    bike_model: str = "",
    top_k: int = 10,
    distance_threshold: float | None = None,
) -> list[SearchResult]:
    """Combined vector + full-text search, deduplicated.

    Vector results come first (semantic relevance), then FTS results
    supplement with exact keyword matches that embeddings may miss.
    """
    vector_results = search_manuals(
        query, bike_model=bike_model, top_k=top_k,
        distance_threshold=distance_threshold,
    )
    fts_results = search_manuals_fts(query, bike_model=bike_model, top_k=5)

    # Deduplicate: vector results first, then unique FTS results
    seen = set()
    combined = []

    for r in vector_results:
        key = (r.manual_title, r.page_number, r.text[:100])
        seen.add(key)
        combined.append(r)

    for r in fts_results:
        key = (r.manual_title, r.page_number, r.text[:100])
        if key not in seen:
            seen.add(key)
            combined.append(r)

    return combined[:top_k]


def _arrow_table_to_dicts(arrow_table) -> list[dict]:
    """Convert a pyarrow Table to a list of dicts without pandas."""
    columns = arrow_table.column_names
    arrays = {col: arrow_table.column(col).to_pylist() for col in columns}
    return [
        {col: arrays[col][i] for col in columns}
        for i in range(arrow_table.num_rows)
    ]


def get_page_content(manual_title: str, page_number: int) -> list[SearchResult]:
    """Get all content from a specific page of a manual."""
    db = _get_db()

    if "manuals" not in db.table_names():
        return []

    table = db.open_table("manuals")
    arrow = table.to_arrow()
    rows = _arrow_table_to_dicts(arrow)

    return [
        SearchResult(
            text=r["text"],
            page_number=r["page_number"],
            content_type=r["content_type"],
            section=r.get("section", ""),
            manual_title=r.get("manual_title", ""),
            bike_model=r.get("bike_model", ""),
            model_year=r.get("model_year"),
            score=0.0,
        )
        for r in rows
        if r["manual_title"] == manual_title and r["page_number"] == page_number
    ]


def list_ingested_manuals() -> list[dict]:
    """List all ingested manuals with summary info."""
    db = _get_db()

    if "manuals" not in db.table_names():
        return []

    table = db.open_table("manuals")
    arrow = table.to_arrow()
    rows = _arrow_table_to_dicts(arrow)

    if not rows:
        return []

    manuals = {}
    for row in rows:
        title = row["manual_title"]
        if title not in manuals:
            manuals[title] = {
                "manual_title": title,
                "bike_model": row.get("bike_model", "Unknown"),
                "model_year": row.get("model_year"),
                "pages": set(),
                "chunk_count": 0,
            }
        manuals[title]["pages"].add(row["page_number"])
        manuals[title]["chunk_count"] += 1

    result = []
    for info in manuals.values():
        pages = sorted(info["pages"])
        result.append(
            {
                "manual_title": info["manual_title"],
                "bike_model": info["bike_model"],
                "model_year": info["model_year"],
                "page_range": f"{min(pages)}-{max(pages)}",
                "chunk_count": info["chunk_count"],
            }
        )

    return result
