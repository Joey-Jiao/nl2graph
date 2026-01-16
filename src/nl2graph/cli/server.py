import time
import subprocess
from pathlib import Path
from typing import Optional

import typer

from ..base import get_context, ConfigService

server_app = typer.Typer(help="Manage graph database servers")


@server_app.command()
def start(
    dataset: str = typer.Argument(..., help="Dataset name"),
    lang: str = typer.Option(..., "--lang", "-l", help="Query language: cypher, sparql, gremlin"),
    timeout: int = typer.Option(60, "--timeout", "-t", help="Startup timeout in seconds"),
):
    """Start a graph database server."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)

    connection = config.get(f"data.{dataset}.connection.{lang}")
    if not connection:
        typer.echo(f"Error: No connection config for {dataset}/{lang}", err=True)
        raise typer.Exit(1)

    if lang == "sparql":
        typer.echo("SPARQL uses local RDFLib, no server needed.")
        return

    if lang == "gremlin":
        _start_gremlin(config, dataset, connection, timeout)
    elif lang == "cypher":
        typer.echo("Neo4j server management not implemented. Start manually.")
        typer.echo(f"  Host: {connection.get('host')}:{connection.get('port')}")
    else:
        typer.echo(f"Error: Unsupported language: {lang}", err=True)
        raise typer.Exit(1)


@server_app.command()
def stop(
    dataset: str = typer.Argument(..., help="Dataset name"),
    lang: str = typer.Option(..., "--lang", "-l", help="Query language: cypher, sparql, gremlin"),
):
    """Stop a graph database server."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)

    connection = config.get(f"data.{dataset}.connection.{lang}")
    if not connection:
        typer.echo(f"Error: No connection config for {dataset}/{lang}", err=True)
        raise typer.Exit(1)

    if lang == "sparql":
        typer.echo("SPARQL uses local RDFLib, no server to stop.")
        return

    if lang == "gremlin":
        _stop_gremlin(config, dataset, connection)
    elif lang == "cypher":
        typer.echo("Neo4j server management not implemented. Stop manually.")
    else:
        typer.echo(f"Error: Unsupported language: {lang}", err=True)
        raise typer.Exit(1)


@server_app.command()
def status(
    dataset: str = typer.Argument(..., help="Dataset name"),
    lang: str = typer.Option(..., "--lang", "-l", help="Query language: cypher, sparql, gremlin"),
):
    """Check server status."""
    ctx = get_context()
    config = ctx.resolve(ConfigService)

    connection = config.get(f"data.{dataset}.connection.{lang}")
    if not connection:
        typer.echo(f"Error: No connection config for {dataset}/{lang}", err=True)
        raise typer.Exit(1)

    if lang == "sparql":
        data_path = connection.get("data_path")
        if data_path and Path(data_path).exists():
            typer.echo(f"SPARQL: TTL file exists at {data_path}")
        else:
            typer.echo(f"SPARQL: TTL file not found at {data_path}")
        return

    host = connection.get("host", "localhost")
    port = connection.get("port")

    if _check_port(host, port):
        typer.echo(f"{lang}: Running at {host}:{port}")
    else:
        typer.echo(f"{lang}: Not running (port {port} closed)")


def _start_gremlin(config: ConfigService, dataset: str, connection: dict, timeout: int):
    docker_compose = connection.get("docker_compose")
    if not docker_compose:
        docker_compose = f"data/{dataset}/server/gremlin/docker-compose.yml"

    docker_compose_path = Path(docker_compose)
    if not docker_compose_path.exists():
        typer.echo(f"Error: docker-compose.yml not found at {docker_compose}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting JanusGraph for {dataset}...")

    result = subprocess.run(
        ["docker", "compose", "-f", str(docker_compose_path), "up", "-d"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        typer.echo(f"Error: Failed to start container", err=True)
        typer.echo(result.stderr, err=True)
        raise typer.Exit(1)

    host = connection.get("host", "localhost")
    port = connection.get("port", 8182)

    typer.echo(f"Waiting for JanusGraph to be ready on port {port}...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        if _check_port(host, port):
            typer.echo("JanusGraph is ready.")
            _load_gremlin_data(config, dataset, connection)
            return
        time.sleep(2)

    typer.echo(f"Error: Timeout waiting for JanusGraph", err=True)
    raise typer.Exit(1)


def _stop_gremlin(config: ConfigService, dataset: str, connection: dict):
    docker_compose = connection.get("docker_compose")
    if not docker_compose:
        docker_compose = f"data/{dataset}/server/gremlin/docker-compose.yml"

    docker_compose_path = Path(docker_compose)
    if not docker_compose_path.exists():
        typer.echo(f"Error: docker-compose.yml not found at {docker_compose}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Stopping JanusGraph for {dataset}...")

    result = subprocess.run(
        ["docker", "compose", "-f", str(docker_compose_path), "down"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        typer.echo(f"Error: Failed to stop container", err=True)
        typer.echo(result.stderr, err=True)
        raise typer.Exit(1)

    typer.echo("JanusGraph stopped.")


def _load_gremlin_data(config: ConfigService, dataset: str, connection: dict):
    from gremlin_python.driver.client import Client

    script_path = connection.get("load_script")
    if not script_path:
        script_path = f"data/{dataset}/server/gremlin/load.groovy"

    script_file = Path(script_path)
    if not script_file.exists():
        typer.echo(f"Warning: Load script not found at {script_path}", err=True)
        return

    host = connection.get("host", "localhost")
    port = connection.get("port", 8182)

    typer.echo("Loading data into JanusGraph...")

    try:
        client = Client(f"ws://{host}:{port}/gremlin", "g")
        script_content = script_file.read_text()
        result = client.submit(script_content).all().result()
        client.close()

        typer.echo("Data loaded successfully.")
        for item in result:
            typer.echo(f"  {item}")
    except Exception as e:
        typer.echo(f"Warning: Failed to load data: {e}", err=True)


def _check_port(host: str, port: int) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False
