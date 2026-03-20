"""MCP server exposing RAG-based manual search tools."""

from mcp.server.fastmcp import FastMCP

from bike_mechanic.search import get_page_content, list_ingested_manuals, search_manuals

mcp = FastMCP("bike-mechanic-rag")


@mcp.tool()
def search_manual(
    query: str,
    bike_model: str = "",
    content_type: str = "",
    top_k: int = 5,
) -> str:
    """Search the motorcycle service manual knowledge base.

    Args:
        query: The search query (e.g., "cam chain tensioner torque")
        bike_model: Filter by bike model (e.g., "KTM 890 Adventure R")
        content_type: Filter by content type: "table", "text", or "" for all
        top_k: Number of results to return
    """

    results = search_manuals(
        query=query,
        bike_model=bike_model,
        content_type=content_type,
        top_k=top_k,
    )

    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Result {i}] Page {r.page_number} | {r.content_type} | {r.section}\n"
            f"Manual: {r.manual_title} | Model: {r.bike_model}\n"
            f"{r.text}\n"
        )

    return "\n---\n".join(parts)


@mcp.tool()
def get_page(manual_title: str, page_number: int) -> str:
    """Get all content from a specific page of a manual.

    Args:
        manual_title: The manual title/filename (without .pdf)
        page_number: The page number to retrieve
    """

    results = get_page_content(manual_title, page_number)

    if not results:
        return f"No content found for page {page_number} in {manual_title}"

    parts = [f"Page {page_number} of {manual_title}:\n"]
    for r in results:
        parts.append(f"[{r.content_type}] {r.section}\n{r.text}\n")

    return "\n".join(parts)


@mcp.tool()
def list_manuals() -> str:
    """List all ingested service manuals with their metadata."""
    items = list_ingested_manuals()

    if not items:
        return "No manuals ingested yet. Run: bike-mechanic ingest --all"

    parts = []
    for m in items:
        parts.append(
            f"- {m['manual_title']}\n"
            f"  Model: {m['bike_model']}\n"
            f"  Year: {m['model_year']}\n"
            f"  Pages: {m['page_range']}\n"
            f"  Chunks: {m['chunk_count']}"
        )

    return "\n".join(parts)


if __name__ == "__main__":
    mcp.run()
