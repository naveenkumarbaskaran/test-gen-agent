"""Command-line interface for test-gen-agent."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from .agent import TestGenAgent

console = Console()


@click.group()
@click.version_option(package_name="test-gen-agent")
def cli() -> None:
    """test-gen-agent: AI-powered pytest test suite generator."""


@cli.command("generate")
@click.argument(
    "source",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--framework",
    default="pytest",
    show_default=True,
    type=click.Choice(["pytest"]),
    help="Test framework to use.",
)
@click.option(
    "--output",
    "-o",
    default="tests/",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output directory for generated test files.",
)
@click.option(
    "--coverage-target",
    default=80,
    show_default=True,
    type=click.IntRange(0, 100),
    help="Minimum coverage percentage to aim for.",
)
@click.option(
    "--print",
    "print_output",
    is_flag=True,
    default=False,
    help="Print generated tests to stdout instead of writing files.",
)
@click.option(
    "--api-key",
    envvar="ANTHROPIC_API_KEY",
    default=None,
    help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var).",
)
def generate(
    source: Path,
    framework: str,
    output: Path,
    coverage_target: int,
    print_output: bool,
    api_key: str | None,
) -> None:
    """Generate pytest tests for SOURCE (a file or directory).

    SOURCE can be a single .py file or a directory. When a directory is given
    every .py file found recursively (excluding __init__.py and files that start
    with 'test_') will be processed.
    """
    paths = _collect_sources(source)
    if not paths:
        console.print("[yellow]No Python source files found.[/yellow]")
        sys.exit(0)

    agent = TestGenAgent(api_key=api_key)

    for path in paths:
        _generate_one(agent, path, output, framework, coverage_target, print_output)


def _collect_sources(source: Path) -> list[Path]:
    """Return a list of Python source files to process."""
    if source.is_file():
        return [source]

    # Directory: walk recursively
    return [
        p
        for p in sorted(source.rglob("*.py"))
        if p.name != "__init__.py" and not p.name.startswith("test_")
    ]


def _generate_one(
    agent: TestGenAgent,
    source_path: Path,
    output_dir: Path,
    framework: str,
    coverage_target: int,
    print_output: bool,
) -> None:
    """Generate tests for a single source file."""
    console.print(f"\n[bold blue]Generating tests for:[/bold blue] {source_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Thinking...", total=None)
        try:
            test_code = agent.generate_tests(
                source_path=source_path,
                framework=framework,
                coverage_target=coverage_target,
            )
        except FileNotFoundError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Unexpected error:[/red] {exc}")
            raise

    if print_output:
        syntax = Syntax(test_code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title=f"Tests for {source_path.name}"))
        return

    # Determine output path
    output_dir.mkdir(parents=True, exist_ok=True)
    test_filename = f"test_{source_path.stem}.py"
    out_path = output_dir / test_filename
    out_path.write_text(test_code, encoding="utf-8")
    console.print(f"[green]Written:[/green] {out_path}")
    console.print(
        f"[dim]Coverage target: {coverage_target}% | Framework: {framework}[/dim]"
    )


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
