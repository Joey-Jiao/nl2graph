import typer

from .init import init, export
from .ls import ls
from .generate import generate
from .execute import execute
from .evaluate import evaluate
from .train import train
from .report import report

app = typer.Typer(
    name="nl2graph",
    help="nl to graph-query CLI",
    no_args_is_help=True,
)

app.command()(init)
app.command()(export)
app.command()(ls)
app.command()(generate)
app.command()(execute)
app.command()(evaluate)
app.command()(train)
app.command()(report)
