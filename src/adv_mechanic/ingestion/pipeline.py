"""Ingestion pipeline: PDF -> chunks -> LanceDB."""

import hashlib
import re
from pathlib import Path

import lancedb
from sentence_transformers import SentenceTransformer

from adv_mechanic.config import EMBEDDING_MODEL, MANUALS_DIR, VECTORSTORE_DIR
from adv_mechanic.ingestion.chunker import create_chunks
from adv_mechanic.ingestion.pdf_parser import parse_pdf

# Filename-to-model mapping
BIKE_MODEL_PATTERNS = {
    r"890": "KTM 890 Adventure R",
    r"te.?300": "Husqvarna TE 300",
    r"te.?250": "Husqvarna TE 250",
}


def _detect_bike_info(filename: str) -> tuple[str, int | None]:
    """Detect bike model and year from a filename."""
    lower = filename.lower()

    bike_model = ""
    for pattern, model in BIKE_MODEL_PATTERNS.items():
        if re.search(pattern, lower):
            bike_model = model
            break

    model_year = None
    year_match = re.search(r"(20\d{2})", filename)
    if year_match:
        model_year = int(year_match.group(1))

    return bike_model, model_year


def _chunk_hash(text: str, page: int, manual: str) -> str:
    """Compute a hash for deduplication."""
    content = f"{text}:{page}:{manual}"
    return hashlib.md5(content.encode()).hexdigest()


def ingest_manual(
    pdf_path: Path,
    bike_model: str = "",
    model_year: int | None = None,
) -> int:
    """Ingest a single PDF manual into LanceDB.

    Returns the number of chunks stored.
    """
    if not bike_model:
        bike_model, detected_year = _detect_bike_info(pdf_path.name)
        if not model_year:
            model_year = detected_year

    print(f"Parsing: {pdf_path.name}")
    print(f"  Model: {bike_model or 'unknown'}, Year: {model_year or 'unknown'}")

    # Extract and chunk
    contents = parse_pdf(pdf_path, bike_model=bike_model, model_year=model_year)
    table_count = sum(1 for c in contents if c.content_type == "table")
    print(f"  Extracted {len(contents)} blocks ({table_count} tables)")

    chunks = create_chunks(contents)
    print(f"  Created {len(chunks)} chunks")

    if not chunks:
        print("  No chunks to ingest.")
        return 0

    # Embed
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c.text for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Build records
    records = []
    for chunk, embedding in zip(chunks, embeddings):
        records.append(
            {
                "text": chunk.text,
                "vector": embedding.tolist(),
                "chunk_hash": _chunk_hash(
                    chunk.text,
                    chunk.metadata["page_number"],
                    chunk.metadata["manual_title"],
                ),
                **chunk.metadata,
            }
        )

    # Store in LanceDB
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(VECTORSTORE_DIR))

    table_name = "manuals"
    manual_title = pdf_path.stem

    if table_name in db.table_names():
        table = db.open_table(table_name)
        # Remove old chunks from this manual (re-ingestion support)
        try:
            table.delete(f'manual_title = "{manual_title.replace(chr(34), "")}"')
        except Exception:
            pass
        table.add(records)
    else:
        table = db.create_table(table_name, records)

    # Build full-text search index for exact spec lookups
    try:
        table.create_fts_index("text", replace=True)
        print(f"  Built FTS index")
    except Exception as e:
        print(f"  FTS index skipped: {e}")

    print(f"  Stored {len(records)} chunks in LanceDB")
    return len(records)


def ingest_all_manuals(manuals_dir: Path | None = None) -> dict[str, int]:
    """Ingest all PDFs from the manuals directory.

    Returns {filename: chunk_count}.
    """
    if manuals_dir is None:
        manuals_dir = MANUALS_DIR

    pdf_files = sorted(manuals_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {manuals_dir}")
        return {}

    print(f"Found {len(pdf_files)} PDF(s) to ingest\n")

    results = {}
    for pdf_path in pdf_files:
        count = ingest_manual(pdf_path)
        results[pdf_path.name] = count
        print()

    total = sum(results.values())
    print(f"Done: {total} total chunks from {len(results)} manual(s)")
    return results
