from typing import Optional
from pathlib import Path

import typer

from ..base.context import get_context
from ..base.configs import ConfigService
from ..execution.service import GraphService
from ..execution import Execution
from ..data.repository import SourceRepository, ResultRepository
from ..pipeline.inference import InferencePipeline, IfExists


def execute(
    dataset: str = typer.Argument(..., help="Dataset name"),
    method: str = typer.Option(..., "--method", "-m", help="Generation method: llm or seq2seq"),
    model: str = typer.Option(..., "--model", help="Model name"),
    lang: str = typer.Option(..., "--lang", "-l", help="Query language: cypher, sparql, kopl"),
    hop: Optional[int] = typer.Option(None, "--hop", help="Filter by hop"),
    split: Optional[str] = typer.Option(None, "--split", help="Filter by split"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers"),
    if_exists: IfExists = typer.Option("skip", "--if-exists", help="Action when record exists: skip or override"),
):
    """Execute generated queries against database."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)

    src_path = config.get(f"data.{dataset}.src")
    dst_path = config.get(f"data.{dataset}.dst")

    if not src_path or not Path(src_path).exists():
        typer.echo(f"Error: src.db not found.", err=True)
        raise typer.Exit(1)

    if not dst_path or not Path(dst_path).exists():
        typer.echo(f"Error: dst.db not found. Run 'nl2graph generate' first.", err=True)
        raise typer.Exit(1)

    try:
        graph_service = ctx.resolve(GraphService)
        connector = graph_service.get_connector(dataset, lang)
        execution = Execution(connector)
    except Exception as e:
        typer.echo(f"Error: Failed to connect to database: {e}", err=True)
        raise typer.Exit(1)

    with SourceRepository(src_path) as src, ResultRepository(dst_path) as dst:
        records = _load_records(src, hop, split)
        typer.echo(f"Executing for {len(records)} records...")

        pipeline = InferencePipeline(
            generator=_DummyGenerator(),
            dst=dst,
            execution=execution,
            method=method,
            lang=lang,
            model=model,
            workers=workers,
            if_exists=if_exists,
        )

        pipeline.execute(records)

    typer.echo("Done.")


class _DummyGenerator:
    def generate(self, text: str) -> str:
        return ""


def _load_records(src: SourceRepository, hop: Optional[int], split: Optional[str]):
    filters = {}
    if hop is not None:
        filters["hop"] = hop
    if split is not None:
        filters["split"] = split

    if filters:
        return list(src.iter_by_filter(**filters))
    return list(src.iter_all())
