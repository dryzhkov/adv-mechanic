"""PDF parser with table-aware extraction and watermark filtering."""

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass
class ExtractedContent:
    """A piece of content extracted from a PDF page."""

    text: str
    page_number: int
    content_type: str  # "table" or "text"
    section: str = ""
    manual_title: str = ""
    bike_model: str = ""
    model_year: int | None = None


# Fonts known to be watermark/annotation layers in KTM/Husqvarna manuals
_WATERMARK_FONTS = {"ProximaNova-Regular-Identity-H"}
_ANNOTATION_FONTS_PARTIAL = {"Circle Frame"}


def _is_watermark_char(obj: dict) -> bool:
    """Check if a PDF char object is part of a watermark or annotation."""
    if obj.get("object_type") != "char":
        return False

    # Non-upright text (rotated watermark along page edge)
    if not obj.get("upright", True):
        return True

    fontname = obj.get("fontname", "")

    # Known watermark font
    if fontname in _WATERMARK_FONTS:
        return True

    # Diagram callout markers (circled numbers in exploded views)
    if any(af in fontname for af in _ANNOTATION_FONTS_PARTIAL):
        return True

    # Large Arial chars used as vertical watermark letters down the margin
    if fontname == "Arial-BoldMT" and obj.get("size", 0) > 12:
        return True

    return False


def _filter_page(page) -> "pdfplumber.Page":
    """Return a filtered page with watermark/annotation chars removed."""
    return page.filter(lambda obj: not _is_watermark_char(obj))


def _is_data_table(table: list[list[str | None]]) -> bool:
    """Check if a table contains actual data (not just figure references)."""
    if not table or len(table) < 2:
        return False

    fig_ref_pattern = re.compile(r"^[A-Z]?\d{5}-\d+$")
    non_empty_cells = 0
    fig_ref_cells = 0

    for row in table:
        for cell in row:
            text = (cell or "").strip()
            if text:
                non_empty_cells += 1
                if fig_ref_pattern.match(text):
                    fig_ref_cells += 1

    # Skip tables that are mostly figure references or too sparse
    if non_empty_cells < 3:
        return False
    if non_empty_cells > 0 and fig_ref_cells / non_empty_cells > 0.5:
        return False

    return True


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber table to markdown format."""
    if not table or len(table) < 2:
        return ""

    # Clean cells
    cleaned = []
    for row in table:
        cleaned.append([(cell or "").strip() for cell in row])

    header = cleaned[0]
    col_count = len(header)

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]

    for row in cleaned[1:]:
        padded = row[:col_count]
        while len(padded) < col_count:
            padded.append("")
        lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines)


def parse_pdf(
    pdf_path: Path,
    bike_model: str = "",
    model_year: int | None = None,
) -> list[ExtractedContent]:
    """Parse a PDF and extract content with table awareness and watermark filtering.

    Watermark characters (rotated text, large margin letters, annotation fonts)
    are filtered out before extraction. Tables are extracted as markdown and
    kept as atomic units. Figure-reference-only tables are skipped.
    """
    contents = []
    manual_title = pdf_path.stem
    current_section = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Filter watermark characters from the page
            clean_page = _filter_page(page)

            # Extract tables as atomic chunks (from filtered page)
            for table in clean_page.extract_tables():
                if not _is_data_table(table):
                    continue
                md = _table_to_markdown(table)
                if md:
                    contents.append(
                        ExtractedContent(
                            text=md,
                            page_number=page_num,
                            content_type="table",
                            section=current_section,
                            manual_title=manual_title,
                            bike_model=bike_model,
                            model_year=model_year,
                        )
                    )

            # Extract page text (from filtered page)
            text = clean_page.extract_text() or ""
            if not text.strip():
                continue

            # Detect section headers (short uppercase lines)
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped and len(stripped) < 80 and stripped.isupper():
                    current_section = stripped

            contents.append(
                ExtractedContent(
                    text=text,
                    page_number=page_num,
                    content_type="text",
                    section=current_section,
                    manual_title=manual_title,
                    bike_model=bike_model,
                    model_year=model_year,
                )
            )

    return contents
