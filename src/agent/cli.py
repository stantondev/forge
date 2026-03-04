"""
Forge — Interactive CLI
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from config import AppConfig
from src.agent.core import Agent


def print_banner(console: Console):
    console.print()
    console.print(Panel(
        "[bold]Forge[/bold]\n"
        "Privacy-first AI with enhanced reasoning\n\n"
        "[dim]Commands:[/dim]\n"
        "  /quick    — Fast single-pass answer (skip reasoning)\n"
        "  /reason   — Full reasoning pipeline (default)\n"
        "  /cloud    — Toggle cloud GPU burst mode\n"
        "  /spend    — Show cloud spending this month\n"
        "  /index    — Index a directory for RAG\n"
        "  /model    — Show current model info\n"
        "  /config   — Show current configuration\n"
        "  /quit     — Exit",
        border_style="blue",
    ))
    console.print()


def handle_command(command: str, agent: Agent, console: Console, config: AppConfig) -> bool:
    """Handle slash commands. Returns True if the command was handled."""
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if cmd in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye.[/dim]")
        sys.exit(0)

    elif cmd == "/model":
        console.print(Panel(
            f"Reasoning model: {config.model.reasoning_model}\n"
            f"Embedding model: {config.model.embedding_model}\n"
            f"Ollama URL: {config.model.ollama_base_url}",
            title="Model Configuration",
        ))
        return True

    elif cmd == "/config":
        console.print(Panel(
            f"Consistency samples: {config.reasoning.consistency_samples}\n"
            f"Reflection enabled: {config.reasoning.enable_reflection}\n"
            f"Max reflection rounds: {config.reasoning.max_reflection_rounds}\n"
            f"Show reasoning: {config.reasoning.show_reasoning}\n"
            f"RAG top-k: {config.rag.top_k}\n"
            f"Chunk size: {config.rag.chunk_size}\n"
            f"Cloud enabled: {config.cloud.enabled}\n"
            f"Cloud budget: ${config.cloud.monthly_budget_limit:.2f}/month",
            title="Configuration",
        ))
        return True

    elif cmd == "/cloud":
        if not config.cloud.enabled:
            console.print(
                "[yellow]Cloud burst not configured.[/yellow]\n"
                "[dim]Set cloud.enabled = True in config/settings.py\n"
                "and set FORGE_CLOUD_API_KEY environment variable.[/dim]"
            )
            return True
        if agent._using_cloud:
            agent.use_cloud(False)
            console.print("[green]Switched to local inference (free)[/green]")
        else:
            console.print("[dim]Connecting to cloud GPU...[/dim]")
            if agent.use_cloud(True):
                console.print("[green]Cloud GPU active — faster inference[/green]")
            else:
                console.print("[red]Cloud unavailable (check API key or budget)[/red]")
        return True

    elif cmd == "/spend":
        manager = agent._get_cloud_manager()
        if manager:
            status = manager.get_status()
            console.print(Panel(
                f"Provider: {status['provider']}\n"
                f"Instance running: {'Yes' if status['instance_running'] else 'No'}\n"
                f"Spent this month: ${status['monthly_spend']:.2f}\n"
                f"Budget limit: ${status['budget_limit']:.2f}\n"
                f"Remaining: ${status['budget_remaining']:.2f}",
                title="Cloud Spending",
                border_style="cyan",
            ))
        else:
            console.print("[dim]Cloud not configured.[/dim]")
        return True

    elif cmd == "/index":
        if not arg:
            console.print("[red]Usage: /index /path/to/documents[/red]")
            return True
        from src.rag.indexer import index_directory
        try:
            count = index_directory(arg, config)
            console.print(f"[green]Indexed {count} chunks from {arg}[/green]")
            # Reset query engine so it picks up new documents
            agent._query_engine = None
        except FileNotFoundError:
            console.print(f"[red]Directory not found: {arg}[/red]")
        except Exception as e:
            console.print(f"[red]Error indexing: {e}[/red]")
        return True

    return False


def main():
    console = Console()
    config = AppConfig()

    print_banner(console)

    console.print("[dim]Initializing agent...[/dim]")
    try:
        agent = Agent(config)
    except Exception as e:
        console.print(f"[red]Failed to initialize agent: {e}[/red]")
        console.print("[dim]Make sure Ollama is running: ollama serve[/dim]")
        sys.exit(1)
    console.print("[green]Ready.[/green]\n")

    # Set up prompt with history
    history_path = Path.home() / ".local-llm" / "history.txt"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(history=FileHistory(str(history_path)))

    use_reasoning = True

    while True:
        try:
            cloud_tag = " cloud" if agent._using_cloud else ""
            mode = f"[{'reasoning' if use_reasoning else 'quick'}{cloud_tag}]"
            user_input = session.prompt(f"{mode} > ").strip()
        except (EOFError, KeyboardInterrupt):
            if agent._using_cloud:
                console.print("\n[dim]Shutting down cloud instance...[/dim]")
                agent.shutdown_cloud()
            console.print("[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # Handle mode toggles
        if user_input.lower() == "/quick":
            use_reasoning = False
            console.print("[dim]Switched to quick mode (single-pass, no reflection)[/dim]")
            continue
        elif user_input.lower() == "/reason":
            use_reasoning = True
            console.print("[dim]Switched to reasoning mode (CoT + consistency + reflection)[/dim]")
            continue

        # Handle other commands
        if user_input.startswith("/"):
            if handle_command(user_input, agent, console, config):
                continue

        # Process the query
        console.print()
        with console.status("[bold blue]Thinking...[/bold blue]"):
            result = agent.query(user_input, use_reasoning=use_reasoning)

        # Display results
        if config.reasoning.show_reasoning and result.get("reasoning") and use_reasoning:
            console.print(Panel(
                Text(result["reasoning"], style="dim"),
                title="Reasoning",
                border_style="dim",
                expand=False,
            ))

        if result.get("critique"):
            console.print(Panel(
                Text(result["critique"], style="yellow"),
                title="Self-Critique",
                border_style="yellow",
                expand=False,
            ))

        if result.get("retrieved_context"):
            console.print(Panel(
                Text(result["retrieved_context"][:500], style="dim"),
                title="Retrieved Context",
                border_style="cyan",
                expand=False,
            ))

        # Main answer
        meta_parts = []
        if result.get("query_type"):
            meta_parts.append(f"type: {result['query_type']}")
        if result.get("agreement_rate") is not None:
            meta_parts.append(f"agreement: {result['agreement_rate']:.0%}")
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""

        console.print(Panel(
            Markdown(result["answer"]),
            title=f"Answer{meta}",
            border_style="green",
        ))
        console.print()


if __name__ == "__main__":
    main()
