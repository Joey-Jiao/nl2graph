from typing import Optional
from pathlib import Path

import typer

from ..base import get_context, ConfigService, LLMService, ModelService, TemplateService
from ..data.repository import SourceRepository, ResultRepository
from ..execution.schema import load_schema
from ..execution.schema.base import BaseSchema
from ..pipeline.generate import GeneratePipeline, IfExists
from ._helpers import load_records, detect_provider


def generate(
    dataset: str = typer.Argument(..., help="Dataset name"),
    method: str = typer.Option(..., "--method", "-m", help="Generation method: llm or seq2seq"),
    model: str = typer.Option(..., "--model", help="Model name"),
    lang: str = typer.Option(..., "--lang", "-l", help="Query language: cypher, sparql, kopl"),
    ir: bool = typer.Option(False, "--ir", help="Enable IR mode (seq2seq)"),
    hop: Optional[int] = typer.Option(None, "--hop", help="Filter by hop"),
    split: Optional[str] = typer.Option(None, "--split", help="Filter by split"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers"),
    if_exists: IfExists = typer.Option("skip", "--if-exists", help="Action when record exists: skip or override"),
):
    """Generate queries from questions."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)

    src_path = config.get(f"data.{dataset}.src")
    dst_path = config.get(f"data.{dataset}.dst")

    if not src_path or not Path(src_path).exists():
        typer.echo(f"Error: src.db not found. Run 'nl2graph init {dataset}' first.", err=True)
        raise typer.Exit(1)

    schema = None
    template_service = None
    template_name = None
    translator = None

    if method == "llm":
        provider = detect_provider(model)
        if not provider:
            typer.echo(f"Error: Unknown model provider for '{model}'", err=True)
            raise typer.Exit(1)

        template_service = ctx.resolve(TemplateService)
        available = template_service.ls_templates("prompts")
        if lang not in available:
            typer.echo(f"Error: No template for '{lang}'. Available: {available}", err=True)
            raise typer.Exit(1)
        template_name = lang

        schema = _load_schema(config, dataset, lang)
        if schema is None:
            typer.echo(f"Error: No schema configured for '{dataset}/{lang}'", err=True)
            raise typer.Exit(1)

        generator = _create_llm_generator(ctx, provider, model, template_service, template_name)

    elif method == "seq2seq":
        if ir:
            try:
                from graphq_trans import Translator
                translator = Translator()
            except ImportError:
                typer.echo("Error: graphq-trans not installed for IR mode", err=True)
                raise typer.Exit(1)

        generator = _create_seq2seq_generator(ctx, config, model, translator, lang)
        if generator is None:
            raise typer.Exit(1)

    else:
        typer.echo(f"Error: Unknown method '{method}'", err=True)
        raise typer.Exit(1)

    with SourceRepository(src_path) as src, ResultRepository(dst_path) as dst:
        records = load_records(src, hop, split)
        typer.echo(f"Generating for {len(records)} records...")

        pipeline = GeneratePipeline(
            generator=generator,
            dst=dst,
            method=method,
            lang=lang,
            model=model,
            workers=workers,
            if_exists=if_exists,
        )

        pipeline.run(records, schema)

    typer.echo("Done.")


def _create_llm_generator(ctx, provider: str, model: str, template_service: TemplateService, template_name: str):
    llm_service = ctx.resolve(LLMService)
    from ..generation.llm.generation import Generation
    return Generation(
        llm_service=llm_service,
        provider=provider,
        model=model,
        template_service=template_service,
        template_name=template_name,
        extract_query=True,
    )


def _create_seq2seq_generator(ctx, config: ConfigService, model: str, translator, lang: str):
    model_service = ctx.resolve(ModelService)
    checkpoint_config = model_service.get_checkpoint_config(model)
    if not checkpoint_config:
        typer.echo(f"Error: Checkpoint '{model}' not found", err=True)
        return None

    checkpoint_path = model_service.get_checkpoint_path(model)
    if not checkpoint_path or not checkpoint_path.exists():
        typer.echo(f"Error: Checkpoint path not found: {checkpoint_path}", err=True)
        return None

    from ..generation.seq2seq.generation import Generation
    return Generation(
        model_path=str(checkpoint_path),
        config_service=config,
        translator=translator,
        lang=lang,
    )


def _load_schema(config: ConfigService, dataset: str, lang: str) -> Optional[BaseSchema]:
    schema_path = config.get(f"data.{dataset}.schema.{lang}")
    if not schema_path:
        return None
    return load_schema(schema_path, lang)
