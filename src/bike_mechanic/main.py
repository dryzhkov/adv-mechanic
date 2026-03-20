"""CLI entry point for bike-mechanic."""

import sys
import time
from pathlib import Path

import typer

from bike_mechanic.config import MANUALS_DIR
from bike_mechanic.graph import NODE_LABELS, build_graph
from bike_mechanic.ingestion.pipeline import ingest_all_manuals, ingest_manual
from bike_mechanic.search import list_ingested_manuals

app = typer.Typer(help="Bike Mechanic - Agentic RAG for motorcycle service manuals")


def _node_summary(node_name: str, state_update: dict | None) -> str:
    """Return a short summary of what a node produced."""
    if not state_update:
        return ""
    if node_name == "router":
        qt = state_update.get("query_type", "?")
        bm = state_update.get("bike_model", "")
        return f"type={qt}" + (f", bike={bm}" if bm else "")

    if node_name == "retrieve":
        docs = state_update.get("retrieved_docs", [])
        if docs:
            top = docs[0]
            return f"{len(docs)} docs, top: p.{top.page_number} (dist={top.score:.2f})"
        return "0 docs"

    if node_name == "grade":
        grade = state_update.get("retrieval_grade", "?")
        conf = state_update.get("retrieval_confidence", "?")
        return f"grade={grade}, confidence={conf}"

    if node_name == "web_search":
        results = state_update.get("web_results", [])
        return f"{len(results)} results"

    if node_name == "resolve_conflicts":
        conflict = state_update.get("has_conflict", False)
        return f"conflict={'yes' if conflict else 'no'}"

    if node_name == "generate":
        score = state_update.get("confidence_score", 0.0)
        return f"confidence={score:.0%}"

    return ""


def _run_graph(graph, input_state: dict) -> dict:
    """Execute graph with per-node progress output."""
    result = {}
    node_start = time.time()

    for event in graph.stream(input_state):
        for node_name, state_update in event.items():
            elapsed = time.time() - node_start
            label = NODE_LABELS.get(node_name, node_name)
            summary = _node_summary(node_name, state_update)
            line = f"  {label}... done ({elapsed:.1f}s)"
            if summary:
                line += f"  [{summary}]"
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
            if state_update:
                result.update(state_update)
            node_start = time.time()

    return result


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

    typer.echo("Bike Mechanic - Motorcycle Service Manual Q&A")
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

        result = _run_graph(graph, {"query": query, "bike_model": bike})

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
    result = _run_graph(graph, {"query": question, "bike_model": bike})

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
        typer.echo("No manuals ingested yet. Run: bike-mechanic ingest --all")
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
