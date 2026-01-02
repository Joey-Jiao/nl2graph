from typing import Literal
from pathlib import Path

import typer

from ..base.context import get_context
from ..base.configs import ConfigService
from ..data.repository import ResultRepository


def clear(
    dataset: str = typer.Argument(..., help="Dataset name"),
    method: Literal["llm", "seq2seq"] = typer.Option(..., "--method", "-m", help="Generation method"),
    model: str = typer.Option(..., "--model", help="Model name"),
    lang: str = typer.Option(..., "--lang", "-l", help="Query language"),
    stage: Literal["gen", "exec", "eval"] = typer.Option("gen", "--stage", "-s", help="Stage to clear (cascades)"),
):
    """Clear results from dst.db for a specific run."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)

    dst_path = config.get(f"data.{dataset}.dst")
    if not dst_path:
        typer.echo(f"Error: No dst.db path configured for dataset '{dataset}'", err=True)
        raise typer.Exit(1)

    if not Path(dst_path).exists():
        typer.echo(f"Error: dst.db not found: {dst_path}", err=True)
        raise typer.Exit(1)

    cascade_info = {
        "gen": "gen, exec, eval",
        "exec": "exec, eval",
        "eval": "eval",
    }

    typer.echo(f"Clearing {cascade_info[stage]} for {method}/{model}/{lang}...")

    with ResultRepository(dst_path) as dst:
        count = dst.clear_stage(method, lang, model, stage)

    typer.echo(f"Done. Cleared {count} records.")
