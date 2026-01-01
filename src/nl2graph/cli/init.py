from typing import Optional
from pathlib import Path

import typer

from ..base.context import get_context
from ..base.configs import ConfigService
from ..data.repository import SourceRepository, ResultRepository


def init(
    dataset: str = typer.Argument(..., help="Dataset name (metaqa, kqapro, openreview)"),
    json_path: Optional[Path] = typer.Option(None, "--json", "-j", help="Override data.json path"),
):
    """Initialize src.db from data.json."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)

    data_path = json_path or config.get(f"data.{dataset}.eval.data")
    if not data_path:
        typer.echo(f"Error: No data path configured for dataset '{dataset}'", err=True)
        raise typer.Exit(1)

    data_path = Path(data_path)
    if not data_path.exists():
        typer.echo(f"Error: Data file not found: {data_path}", err=True)
        raise typer.Exit(1)

    src_path = config.get(f"data.{dataset}.src")
    if not src_path:
        typer.echo(f"Error: No src.db path configured for dataset '{dataset}'", err=True)
        raise typer.Exit(1)

    typer.echo(f"Initializing {src_path} from {data_path}...")

    with SourceRepository(src_path) as src:
        count = src.init_from_json(str(data_path))

    typer.echo(f"Done. Loaded {count} records.")


def export(
    dataset: str = typer.Argument(..., help="Dataset name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON path"),
):
    """Export dst.db results to JSON."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)

    dst_path = config.get(f"data.{dataset}.dst")
    if not dst_path:
        typer.echo(f"Error: No dst.db path configured for dataset '{dataset}'", err=True)
        raise typer.Exit(1)

    if not Path(dst_path).exists():
        typer.echo(f"Error: dst.db not found: {dst_path}", err=True)
        raise typer.Exit(1)

    output_path = output or Path(f"data/{dataset}/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Exporting {dst_path} to {output_path}...")

    with ResultRepository(dst_path) as dst:
        records = dst.export_json(str(output_path))

    typer.echo(f"Done. Exported {len(records)} results.")
