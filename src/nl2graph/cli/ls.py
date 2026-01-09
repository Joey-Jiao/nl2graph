from typing import Optional

import typer

from ..base import get_context, ConfigService, LLMService, ModelService, TemplateService


def ls(
    resource: str = typer.Argument(
        ...,
        help="Resource type: datasets, models, checkpoints, templates",
    ),
):
    """List available resources."""
    ctx = get_context()

    if resource == "datasets":
        _ls_datasets(ctx)
    elif resource == "models":
        _ls_models(ctx)
    elif resource == "checkpoints":
        _ls_checkpoints(ctx)
    elif resource == "templates":
        _ls_templates(ctx)
    else:
        typer.echo(f"Unknown resource type: {resource}", err=True)
        typer.echo("Available: datasets, models, checkpoints, templates", err=True)
        raise typer.Exit(1)


def _ls_datasets(ctx):
    config = ctx.resolve(ConfigService)
    data_config = config.get("data", {})

    typer.echo("Datasets:")
    for name in data_config:
        if name == "base_dir":
            continue
        src_path = config.get(f"data.{name}.src", "N/A")
        dst_path = config.get(f"data.{name}.dst", "N/A")
        typer.echo(f"  {name}")
        typer.echo(f"    src: {src_path}")
        typer.echo(f"    dst: {dst_path}")


def _ls_models(ctx):
    llm_service = ctx.resolve(LLMService)

    typer.echo("LLM Models:")
    for provider in llm_service.ls_providers():
        typer.echo(f"  {provider}:")
        for model in llm_service.ls_models(provider):
            typer.echo(f"    - {model}")


def _ls_checkpoints(ctx):
    model_service = ctx.resolve(ModelService)

    typer.echo("Seq2Seq Checkpoints:")
    for name in model_service.ls_checkpoints():
        config = model_service.get_checkpoint_config(name)
        path = config.get("path", "N/A") if config else "N/A"
        typer.echo(f"  {name}: {path}")


def _ls_templates(ctx):
    template_service = ctx.resolve(TemplateService)

    typer.echo("Templates:")
    for category in template_service.ls_categories():
        typer.echo(f"  {category}:")
        for name in template_service.ls_templates(category):
            typer.echo(f"    - {name}")
