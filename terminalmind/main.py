from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from openai import APIError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from terminalmind.config import get_settings
from terminalmind.core.agent import ResearchAgent
from terminalmind.core.schemas import ResearchAnswer
from terminalmind.utils.storage import Storage

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True, help="TerminalMind research CLI")


@app.callback()
def _root(
    ctx: typer.Context,
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Override data directory"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable DEBUG logging"),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir
    ctx.obj["verbose"] = verbose


def _build_agent(ctx: typer.Context, require_api_key: bool = True) -> ResearchAgent:
    try:
        settings = get_settings()
    except ValidationError:
        console.print(
            "[red]Missing OPENAI_API_KEY. Copy .env.example to .env and set your key.[/red]"
        )
        raise typer.Exit(1) from None

    overrides: dict[str, object] = {}
    if ctx.obj.get("data_dir") is not None:
        overrides["data_dir"] = ctx.obj["data_dir"]
    if ctx.obj.get("verbose"):
        overrides["log_level"] = "DEBUG"
    if overrides:
        settings = settings.model_copy(update=overrides)

    storage = Storage(settings.data_dir)
    storage.ensure_layout()
    storage.setup_logging(settings.log_level)

    if require_api_key and not settings.openai_api_key:
        console.print(
            "[red]Missing OPENAI_API_KEY. Copy .env.example to .env and set your key.[/red]"
        )
        raise typer.Exit(1)

    return ResearchAgent(settings=settings, storage=storage)


def _render_answer(answer: ResearchAnswer) -> None:
    console.print(Panel(answer.summary, title="Summary", border_style="cyan"))
    keypoints = "\n".join(f"- {p}" for p in answer.key_points) or "- (none)"
    followups = "\n".join(f"- {p}" for p in answer.follow_ups) or "- (none)"
    console.print(Markdown(f"## Key Points\n\n{keypoints}\n\n## Follow-ups\n\n{followups}"))


@app.command("ingest")
def ingest_cmd(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=False, help="Path to .txt or .md file"),
) -> None:
    agent = _build_agent(ctx, require_api_key=False)
    try:
        record, created = agent.ingest(path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if created:
        console.print(f"[green]Ingested[/green] {record.source_path} ({record.char_count} chars)")
    else:
        console.print(f"[yellow]Already ingested[/yellow] (hash {record.content_hash[:12]}…)")


@app.command("search")
def search_cmd(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Research query"),
) -> None:
    agent = _build_agent(ctx, require_api_key=True)
    chunks = agent.storage.load_chunks()
    if not chunks:
        console.print(
            "[yellow]No ingested documents found. Falling back to LLM-only answer.[/yellow]"
        )
    try:
        with console.status("Researching…"):
            entry = agent.search(query)
    except (AuthenticationError, RateLimitError, APITimeoutError, APIError) as exc:
        console.print(f"[red]OpenAI API error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    _render_answer(entry.answer)
    console.print(
        f"[dim]Saved history + report ({entry.id})"
        f"{' · grounded in ingest' if entry.used_ingest else ' · LLM-only'}[/dim]"
    )


@app.command("history")
def history_cmd(ctx: typer.Context) -> None:
    agent = _build_agent(ctx, require_api_key=False)
    entries = agent.storage.load_history()
    if not entries:
        console.print("[dim]No research sessions yet.[/dim]")
        return
    table = Table(title="Research History")
    table.add_column("Time", style="cyan")
    table.add_column("Query")
    table.add_column("Summary")
    table.add_column("Context", justify="center")
    for entry in entries:
        summary = entry.answer.summary
        if len(summary) > 60:
            summary = summary[:57] + "..."
        table.add_row(
            entry.created_at.strftime("%Y-%m-%d %H:%M"),
            entry.query,
            summary,
            "Y" if entry.used_ingest else "N",
        )
    console.print(table)


if __name__ == "__main__":
    app()
