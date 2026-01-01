from typing import Optional
from pathlib import Path

import typer

from ..base.context import get_context
from ..base.configs import ConfigService
from ..base.llm.service import LLMService
from ..base.models.service import ModelService
from ..base.templates.service import TemplateService
from ..data.repository import SourceRepository, ResultRepository
from ..execution.schema.property_graph import PropertyGraphSchema
from ..pipeline.inference import InferencePipeline


def generate(
    dataset: str = typer.Argument(..., help="Dataset name"),
    method: str = typer.Option(..., "--method", "-m", help="Generation method: llm or seq2seq"),
    model: str = typer.Option(..., "--model", help="Model name"),
    lang: str = typer.Option(..., "--lang", "-l", help="Query language: cypher, sparql, kopl"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="Prompt template name"),
    ir: bool = typer.Option(False, "--ir", help="Enable IR mode (seq2seq)"),
    hop: Optional[int] = typer.Option(None, "--hop", help="Filter by hop"),
    split: Optional[str] = typer.Option(None, "--split", help="Filter by split"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers"),
):
    """Generate queries from questions."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)

    src_path = config.get(f"data.{dataset}.src")
    dst_path = config.get(f"data.{dataset}.dst")

    if not src_path or not Path(src_path).exists():
        typer.echo(f"Error: src.db not found. Run 'nl2graph init {dataset}' first.", err=True)
        raise typer.Exit(1)

    generator = _create_generator(ctx, config, method, model)
    if generator is None:
        raise typer.Exit(1)

    template_service = None
    if template:
        template_service = ctx.resolve(TemplateService)

    translator = None
    if ir:
        try:
            from graphq_trans import Translator
            translator = Translator()
        except ImportError:
            typer.echo("Error: graphq-trans not installed for IR mode", err=True)
            raise typer.Exit(1)

    schema = _load_schema(config, dataset)

    with SourceRepository(src_path) as src, ResultRepository(dst_path) as dst:
        records = _load_records(src, hop, split)
        typer.echo(f"Generating for {len(records)} records...")

        pipeline = InferencePipeline(
            generator=generator,
            dst=dst,
            template_service=template_service,
            template_name=template,
            translator=translator,
            method=method,
            lang=lang,
            model=model,
            ir_mode=ir,
            extract_query=(method == "llm"),
            workers=workers,
        )

        pipeline.generate(records, schema)

    typer.echo("Done.")


def _create_generator(ctx, config: ConfigService, method: str, model: str):
    if method == "llm":
        llm_service = ctx.resolve(LLMService)
        provider = _detect_provider(model)
        if not provider:
            typer.echo(f"Error: Unknown model provider for '{model}'", err=True)
            return None

        from ..generation.llm.generation import Generation
        return Generation(llm_service, provider, model)

    elif method == "seq2seq":
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
        return Generation(str(checkpoint_path), config)

    else:
        typer.echo(f"Error: Unknown method '{method}'", err=True)
        return None


def _detect_provider(model: str) -> Optional[str]:
    if model.startswith("gpt"):
        return "openai"
    elif model.startswith("deepseek"):
        return "deepseek"
    return None


def _load_schema(config: ConfigService, dataset: str) -> Optional[PropertyGraphSchema]:
    schema_path = config.get(f"data.{dataset}.schema")
    if not schema_path:
        return None

    schema_path = Path(schema_path)
    if not schema_path.exists():
        return None

    import json
    with open(schema_path) as f:
        data = json.load(f)
    return PropertyGraphSchema.from_dict(data)


def _load_records(src: SourceRepository, hop: Optional[int], split: Optional[str]):
    filters = {}
    if hop is not None:
        filters["hop"] = hop
    if split is not None:
        filters["split"] = split

    if filters:
        return list(src.iter_by_filter(**filters))
    return list(src.iter_all())
