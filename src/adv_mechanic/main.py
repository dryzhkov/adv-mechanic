"""CLI entry point for adv-mechanic."""

from pathlib import Path

import typer

from adv_mechanic.config import MANUALS_DIR
from adv_mechanic.graph import build_graph
from adv_mechanic.ingestion.pipeline import ingest_all_manuals, ingest_manual
from adv_mechanic.search import list_ingested_manuals

app = typer.Typer(help="ADV Mechanic - Agentic RAG for motorcycle service manuals")


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


@app.command()
def ingest(
    path: Path = typer.Argument(None, help="Path to a specific PDF to ingest"),
    all_manuals: bool = typer.Option(
        False, "--all", help="Ingest all PDFs in data/manuals/"
    ),
):
    """Ingest service manual PDFs into the vector store."""
    if path is not None:
        if not path.exists():
            typer.echo(f"File not found: {path}")
            raise typer.Exit(1)
        ingest_manual(path)
    elif all_manuals:
        ingest_all_manuals(MANUALS_DIR)
    else:
        typer.echo("Specify a PDF path or use --all to ingest all manuals.")
        raise typer.Exit(1)


@app.command()
def chat(
    bike: str = typer.Option("", help="Default bike model for queries"),
):
    """Start an interactive chat session."""
    graph = build_graph()

    typer.echo("ADV Mechanic - Motorcycle Service Manual Q&A")
    typer.echo("Type your question, or 'quit' to exit.\n")

    if bike:
        typer.echo(f"Default bike: {bike}\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            typer.echo("Goodbye!")
            break

        result = graph.invoke({"query": query, "bike_model": bike})

        score = result.get("confidence_score", 0.0)
        label = _confidence_label(score)
        typer.echo(f"\nMechanic [{label} {score:.0%}]: {result['answer']}\n")

        if result.get("sources"):
            typer.echo("Sources:")
            for src in result["sources"]:
                typer.echo(f"  - {src}")
            typer.echo()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    bike: str = typer.Option("", help="Bike model"),
):
    """Ask a single question (non-interactive)."""
    graph = build_graph()
    result = graph.invoke({"query": question, "bike_model": bike})

    score = result.get("confidence_score", 0.0)
    label = _confidence_label(score)
    typer.echo(f"[Confidence: {label} {score:.0%}]\n")
    typer.echo(result["answer"])

    if result.get("sources"):
        typer.echo("\nSources:")
        for src in result["sources"]:
            typer.echo(f"  - {src}")


@app.command()
def manuals():
    """List all ingested manuals."""
    items = list_ingested_manuals()
    if not items:
        typer.echo("No manuals ingested yet. Run: adv-mechanic ingest --all")
        return

    for m in items:
        typer.echo(
            f"- {m['manual_title']}\n"
            f"  Model: {m['bike_model']}\n"
            f"  Year: {m['model_year']}\n"
            f"  Pages: {m['page_range']}\n"
            f"  Chunks: {m['chunk_count']}"
        )


if __name__ == "__main__":
    app()
