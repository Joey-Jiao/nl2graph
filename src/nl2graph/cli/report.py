from typing import Optional, List
from pathlib import Path
import json

import typer

from ..base.context import get_context
from ..base.configs import ConfigService
from ..data.repository import SourceRepository, ResultRepository
from ..analysis import Reporting


def report(
    dataset: str = typer.Argument(..., help="Dataset name"),
    method: str = typer.Option(..., "--method", "-m", help="Generation method: llm or seq2seq"),
    model: str = typer.Option(..., "--model", help="Model name"),
    lang: str = typer.Option(..., "--lang", "-l", help="Query language"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or markdown"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Generate evaluation report."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)

    src_path = config.get(f"data.{dataset}.src")
    dst_path = config.get(f"data.{dataset}.dst")

    if not src_path or not Path(src_path).exists():
        typer.echo(f"Error: src.db not found: {src_path}", err=True)
        raise typer.Exit(1)

    if not dst_path or not Path(dst_path).exists():
        typer.echo(f"Error: dst.db not found: {dst_path}", err=True)
        raise typer.Exit(1)

    group_by = config.get(f"data.{dataset}.eval.group_by", [])

    with SourceRepository(src_path) as src, ResultRepository(dst_path) as dst:
        pairs = []
        for result in dst.iter_by_config(method, lang, model):
            record = src.get(result.question_id)
            if record:
                pairs.append((record, result))

        if not pairs:
            typer.echo(f"No results found for {method}/{lang}/{model}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Generating report for {len(pairs)} results...")

        reporting = Reporting()
        config_id = f"{dataset}-{method}-{lang}-{model}"
        report_obj = reporting.generate(pairs, config_id, group_by=group_by)

    if format == "json":
        output_content = report_obj.model_dump_json(indent=2)
    else:
        output_content = _format_markdown(report_obj)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(output_content)
        typer.echo(f"Report saved to {output}")
    else:
        typer.echo(output_content)


def _format_markdown(report) -> str:
    lines = [
        f"# Report: {report.run_id}",
        "",
        f"**Total:** {report.total}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Count | {report.summary.count} |",
        f"| Errors | {report.summary.error_count} |",
        f"| Accuracy | {report.summary.accuracy:.4f} |",
        f"| Avg F1 | {report.summary.avg_f1:.4f} |",
        f"| Avg Precision | {report.summary.avg_precision:.4f} |",
        f"| Avg Recall | {report.summary.avg_recall:.4f} |",
        "",
    ]

    if report.by_field:
        lines.append("## By Field")
        lines.append("")

        for field, stats_dict in report.by_field.items():
            lines.append(f"### {field}")
            lines.append("")
            lines.append("| Value | Count | Errors | Accuracy | F1 |")
            lines.append("|-------|-------|--------|----------|-----|")

            for value, stats in stats_dict.items():
                lines.append(
                    f"| {value} | {stats.count} | {stats.error_count} | "
                    f"{stats.accuracy:.4f} | {stats.avg_f1:.4f} |"
                )
            lines.append("")

    if report.errors.total_errors > 0:
        lines.append("## Errors")
        lines.append("")
        lines.append(f"**Total Errors:** {report.errors.total_errors}")
        lines.append("")

        if report.errors.missing_relations:
            lines.append("**Missing Relations:**")
            for rel in report.errors.missing_relations:
                lines.append(f"- {rel}")
            lines.append("")

        if report.errors.error_types:
            lines.append("**Error Types:**")
            for error_type, count in report.errors.error_types.items():
                lines.append(f"- {error_type}: {count}")
            lines.append("")

    return "\n".join(lines)
