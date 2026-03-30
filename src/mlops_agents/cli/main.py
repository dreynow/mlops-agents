"""CLI entry point for mlops-agents.

Commands:
  mlops-agents run pipeline.yaml     Run a pipeline
  mlops-agents audit                 View the audit trail
  mlops-agents status                Show pipeline status
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="mlops-agents",
    help="Agentic MLOps Orchestration - AI agents that make decisions, not just execute scripts.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    pipeline_path: str = typer.Argument(..., help="Path to pipeline.yaml"),
    entry_stage: Optional[str] = typer.Option(
        None, "--stage", "-s", help="Entry stage (default: first)"
    ),
    max_stages: int = typer.Option(20, "--max-stages", help="Max stages to execute (safety limit)"),
):
    """Run a pipeline from a YAML config file."""
    path = Path(pipeline_path)
    if not path.exists():
        console.print(f"[red]Pipeline file not found: {pipeline_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Running pipeline:[/bold] {path.name}")
    console.print()

    trace = asyncio.run(_run_pipeline(path, entry_stage, max_stages))

    # Print summary
    console.print()
    _print_trace_summary(trace)


async def _run_pipeline(path: Path, entry_stage: str | None, max_stages: int):
    from mlops_agents.core.pipeline import Pipeline

    pipeline = Pipeline.from_yaml(str(path))

    console.print(f"  Pipeline: [cyan]{pipeline.config.name}[/cyan]")
    console.print(f"  Stages: {', '.join(pipeline.config.stages.keys())}")
    console.print(f"  Reasoning: {pipeline.config.reasoning.engine}")
    console.print(f"  Provider: {pipeline.config.provider.backend}")
    console.print()

    trace = await pipeline.run(entry_stage=entry_stage, max_stages=max_stages)
    return trace


def _print_trace_summary(trace):

    status_color = {
        "completed": "green",
        "failed": "red",
        "escalated": "yellow",
        "running": "blue",
    }
    color = status_color.get(trace.status, "white")

    console.print(f"[bold]Trace:[/bold] {trace.trace_id}")
    console.print(f"[bold]Status:[/bold] [{color}]{trace.status}[/{color}]")
    console.print(f"[bold]Decisions:[/bold] {len(trace.decisions)}")
    console.print()

    if trace.decisions:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Agent", style="cyan")
        table.add_column("Action")
        table.add_column("Decision", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Conclusion")

        for d in trace.decisions:
            status = "[green]GO[/green]" if d.approved else "[red]NO-GO[/red]"
            if d.escalate_to_human:
                status = "[yellow]ESCALATED[/yellow]"

            table.add_row(
                d.agent_name,
                d.action,
                status,
                f"{d.reasoning.confidence:.0%}",
                d.reasoning.conclusion[:60],
            )

        console.print(table)


@app.command()
def audit(
    trace_id: Optional[str] = typer.Option(None, "--trace", "-t", help="Specific trace ID"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter by agent name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent decisions"),
    db_path: str = typer.Option("mlops_audit.db", "--db", help="Audit database path"),
):
    """View the audit trail."""
    asyncio.run(_show_audit(trace_id, agent, limit, db_path))


async def _show_audit(trace_id: str | None, agent: str | None, limit: int, db_path: str):
    from mlops_agents.core.audit import SQLiteAuditStore

    store = SQLiteAuditStore(db_path=db_path)

    if trace_id:
        decisions = await store.get_trace(trace_id)
        if not decisions:
            console.print(f"[yellow]No decisions found for trace: {trace_id}[/yellow]")
            return
        console.print(f"\n[bold]Trace:[/bold] {trace_id}")
    elif agent:
        decisions = await store.get_decisions_by_agent(agent, limit=limit)
        if not decisions:
            console.print(f"[yellow]No decisions found for agent: {agent}[/yellow]")
            return
        console.print(f"\n[bold]Agent:[/bold] {agent}")
    else:
        decisions = await store.get_recent(limit=limit)
        if not decisions:
            console.print("[yellow]No decisions in audit trail.[/yellow]")
            return
        console.print("\n[bold]Recent decisions:[/bold]")

    console.print(f"  Showing {len(decisions)} decision(s)\n")

    for d in decisions:
        status = "[green]GO[/green]" if d.approved else "[red]NO-GO[/red]"
        if d.escalate_to_human:
            status = "[yellow]ESCALATED[/yellow]"

        console.print(
            f"  [{d.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"[cyan]{d.agent_name}[/cyan] {d.action} -> {status} "
            f"(confidence: {d.reasoning.confidence:.0%})"
        )
        console.print(f"    Conclusion: {d.reasoning.conclusion}")

        if d.reasoning.observations:
            for obs in d.reasoning.observations[:3]:
                console.print(f"    - {obs}")

        if d.escalate_to_human:
            console.print(f"    [yellow]Escalation: {d.escalation_reason}[/yellow]")
        console.print()


@app.command()
def status(
    db_path: str = typer.Option("mlops_audit.db", "--db", help="Audit database path"),
):
    """Show recent pipeline runs and their status."""
    asyncio.run(_show_status(db_path))


async def _show_status(db_path: str):
    from mlops_agents.core.audit import SQLiteAuditStore

    store = SQLiteAuditStore(db_path=db_path)
    recent = await store.get_recent(limit=50)

    if not recent:
        console.print("[yellow]No pipeline activity found.[/yellow]")
        return

    # Group by trace_id
    traces: dict[str, list] = {}
    for d in recent:
        traces.setdefault(d.trace_id, []).append(d)

    console.print(f"\n[bold]Recent pipeline runs:[/bold] ({len(traces)} traces)\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Trace ID", style="cyan")
    table.add_column("Stages", justify="right")
    table.add_column("Approved", justify="right")
    table.add_column("Rejected", justify="right")
    table.add_column("Escalated", justify="right")
    table.add_column("Last Action")

    for tid, decisions in sorted(traces.items(), key=lambda x: x[1][-1].timestamp, reverse=True)[
        :10
    ]:
        approved = sum(1 for d in decisions if d.approved and not d.escalate_to_human)
        rejected = sum(1 for d in decisions if not d.approved)
        escalated = sum(1 for d in decisions if d.escalate_to_human)
        last = decisions[-1]

        table.add_row(
            tid,
            str(len(decisions)),
            f"[green]{approved}[/green]",
            f"[red]{rejected}[/red]" if rejected else "0",
            f"[yellow]{escalated}[/yellow]" if escalated else "0",
            f"{last.agent_name}/{last.action}",
        )

    console.print(table)


@app.command()
def ingest(
    notebook_path: str = typer.Argument(..., help="Path to Jupyter notebook (.ipynb)"),
    output_dir: str = typer.Option("pipeline", "--output", "-o", help="Output directory"),
):
    """Ingest a Jupyter notebook and generate train.py + pipeline.yaml.

    Supports two modes:
      - Blueprint: notebook has # mlops: tags (deterministic extraction)
      - Inferred: no tags, uses heuristics with confidence levels
    """
    path = Path(notebook_path)
    if not path.exists():
        console.print(f"[red]Notebook not found: {notebook_path}[/red]")
        raise typer.Exit(1)
    if path.suffix != ".ipynb":
        console.print(f"[red]Not a notebook file: {notebook_path}[/red]")
        raise typer.Exit(1)

    from mlops_agents.ingest.generator import generate_all
    from mlops_agents.ingest.parser import SectionType, analyze_notebook

    console.print(f"\n[bold]Ingesting notebook:[/bold] {path.name}")

    structure = analyze_notebook(path)

    # Report mode
    if structure.mode == "blueprint":
        console.print("[green]Blueprint tags detected - deterministic extraction[/green]")
    else:
        console.print("[yellow]No # mlops: tags found. Inferring structure...[/yellow]")

    # Report warnings
    for warning in structure.warnings:
        if warning.startswith("  "):
            console.print(f"  {warning.strip()}")
        else:
            console.print(f"  [yellow]{warning}[/yellow]")

    # Report detections
    if structure.detected_model_type != "unknown":
        console.print(f"  Model type: [cyan]{structure.detected_model_type}[/cyan]")
    if structure.detected_metrics:
        console.print(f"  Metrics: [cyan]{', '.join(structure.detected_metrics)}[/cyan]")

    # Show sections found
    console.print()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Section")
    table.add_column("Cells")
    table.add_column("Confidence", justify="right")

    for section_type in SectionType:
        if section_type == SectionType.UNKNOWN:
            continue
        cells = structure.sections.get(section_type, [])
        if cells:
            avg_conf = sum(c.confidence for c in cells) / len(cells)
            conf_color = "green" if avg_conf >= 0.9 else "yellow" if avg_conf >= 0.8 else "red"
            table.add_row(
                section_type.value,
                ", ".join(str(c.index) for c in cells),
                f"[{conf_color}]{avg_conf:.0%}[/{conf_color}]",
            )
        else:
            table.add_row(section_type.value, "-", "[dim]not found[/dim]")

    console.print(table)

    # Check for missing required sections
    missing = structure.missing_sections
    if missing:
        console.print(
            f"\n[yellow]Missing required sections: {', '.join(s.value for s in missing)}[/yellow]"
        )
        console.print("Add these tags to your notebook:")
        for s in missing:
            console.print(f"  [cyan]# mlops: {s.value}[/cyan]")
        console.print()

    # Generate files
    files = generate_all(structure, output_dir=output_dir)

    console.print(f"\n[green]Generated {len(files)} files in {output_dir}/[/green]")
    for fname in files:
        console.print(f"  {output_dir}/{fname}")

    console.print(
        f"\n[bold]Next steps:[/bold]\n"
        f"  1. Review {output_dir}/train.py (check TODOs)\n"
        f"  2. Adjust {output_dir}/pipeline.yaml thresholds\n"
        f"  3. mlops-agents run {output_dir}/pipeline.yaml"
    )


if __name__ == "__main__":
    app()
