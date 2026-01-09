from typing import Optional
from pathlib import Path

import typer

from ..base import get_context, ConfigService, ModelService
from ..pipeline.train import TrainPipeline


def train(
    dataset: str = typer.Argument(..., help="Dataset name (metaqa, kqapro)"),
    shot: Optional[str] = typer.Option(None, "--shot", "-s", help="Few-shot config (1shot, 3shot, 5shot)"),
    from_checkpoint: Optional[str] = typer.Option(None, "--from", "-f", help="Transfer from checkpoint"),
    preprocess_only: bool = typer.Option(False, "--preprocess-only", "-p", help="Only run preprocessing"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Train seq2seq model: preprocess → train."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)
    model_service = ctx.resolve(ModelService)

    dataset_config_path = _get_dataset_config(config, dataset, shot)
    if not dataset_config_path:
        typer.echo(f"Error: No dataset config found for '{dataset}'", err=True)
        raise typer.Exit(1)

    raw_dir = _get_raw_dir(config, dataset, shot)
    processed_dir = _get_processed_dir(config, dataset, shot)

    if not raw_dir or not Path(raw_dir).exists():
        typer.echo(f"Error: Raw data directory not found: {raw_dir}", err=True)
        raise typer.Exit(1)

    output_dir = output or Path(f"models/checkpoints/{dataset}_{shot or 'default'}")

    model_name_or_path = "facebook/bart-base"
    if from_checkpoint:
        checkpoint_path = model_service.get_checkpoint_path(from_checkpoint)
        if checkpoint_path and checkpoint_path.exists():
            model_name_or_path = str(checkpoint_path)
            typer.echo(f"Transfer learning from: {model_name_or_path}")
        else:
            typer.echo(f"Warning: Checkpoint '{from_checkpoint}' not found, using default model", err=True)

    pipeline = TrainPipeline(config)

    typer.echo(f"Preprocessing: {raw_dir} → {processed_dir}")
    pipeline.preprocess(
        dataset_config_path=dataset_config_path,
        input_dir=Path(raw_dir),
        output_dir=Path(processed_dir),
    )

    if preprocess_only:
        typer.echo("Preprocessing complete.")
        return

    typer.echo(f"Training: {processed_dir} → {output_dir}")
    best_acc = pipeline.train(
        dataset_config_path=dataset_config_path,
        input_dir=Path(processed_dir),
        output_dir=output_dir,
        model_name_or_path=model_name_or_path,
    )

    typer.echo(f"Training complete. Best accuracy: {best_acc:.4f}")


def _get_dataset_config(config: ConfigService, dataset: str, shot: Optional[str]) -> Optional[str]:
    if shot:
        path = config.get(f"data.{dataset}.train.config")
    else:
        path = config.get(f"data.{dataset}.train.config")

    if path and Path(path).exists():
        return path
    return None


def _get_raw_dir(config: ConfigService, dataset: str, shot: Optional[str]) -> Optional[str]:
    if shot:
        return config.get(f"data.{dataset}.train.raw") + f"/{shot}"
    return config.get(f"data.{dataset}.train.raw")


def _get_processed_dir(config: ConfigService, dataset: str, shot: Optional[str]) -> Optional[str]:
    if shot:
        return config.get(f"data.{dataset}.train.shots.{shot}")
    return config.get(f"data.{dataset}.train.processed")
