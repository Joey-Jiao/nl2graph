from typing import Optional
from pathlib import Path

import typer

from ..base import get_context, ConfigService
from ..evaluation import Scoring
from ..data.repository import SourceRepository, ResultRepository
from ..pipeline.evaluate import EvaluatePipeline, IfExists
from ._helpers import load_records


def evaluate(
    dataset: str = typer.Argument(..., help="Dataset name"),
    method: str = typer.Option(..., "--method", "-m", help="Generation method: llm or seq2seq"),
    model: str = typer.Option(..., "--model", help="Model name"),
    lang: str = typer.Option(..., "--lang", "-l", help="Query language: cypher, sparql, kopl"),
    hop: Optional[int] = typer.Option(None, "--hop", help="Filter by hop"),
    split: Optional[str] = typer.Option(None, "--split", help="Filter by split"),
    if_exists: IfExists = typer.Option("skip", "--if-exists", help="Action when record exists: skip or override"),
):
    """Evaluate execution results."""
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

    with SourceRepository(src_path) as src, ResultRepository(dst_path) as dst:
        records = load_records(src, hop, split)
        typer.echo(f"Evaluating {len(records)} records...")

        pipeline = EvaluatePipeline(
            dst=dst,
            method=method,
            lang=lang,
            model=model,
            scoring=Scoring(),
            if_exists=if_exists,
        )

        pipeline.run(records)

    typer.echo("Done.")
