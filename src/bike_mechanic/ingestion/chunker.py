"""Table-aware chunking for service manual content."""

from dataclasses import dataclass, field

from bike_mechanic.ingestion.pdf_parser import ExtractedContent


@dataclass
class Chunk:
    """A chunk ready for embedding and storage."""

    text: str
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int = 800, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, breaking at sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a natural boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size * 0.5:
                    end = start + last_sep + len(sep)
                    break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        start = end - overlap

    return chunks


def create_chunks(
    contents: list[ExtractedContent],
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[Chunk]:
    """Convert extracted content into chunks for embedding.

    Tables are kept as atomic units (never split).
    Text is split with overlap at sentence boundaries.
    """
    chunks = []

    for content in contents:
        base_metadata = {
            "page_number": content.page_number,
            "content_type": content.content_type,
            "section": content.section,
            "manual_title": content.manual_title,
            "bike_model": content.bike_model,
            "model_year": content.model_year,
        }

        if content.content_type == "table":
            # Tables are atomic - never split
            chunks.append(Chunk(text=content.text, metadata=base_metadata))
        else:
            # Split text with overlap
            for text_chunk in _split_text(content.text, chunk_size, overlap):
                chunks.append(
                    Chunk(text=text_chunk, metadata={**base_metadata})
                )

    return chunks
