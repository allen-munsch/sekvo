import sys
import asyncio
from typing import Dict, List, Optional
from functools import wraps
import click
from sekvo.providers.anthropic.generate import AnthropicProvider
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from sekvo.cli.plugin import ProviderPlugin
from sekvo.providers import ProviderRegistry

console = Console()

def async_command(f):
    """Decorator to run async commands"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            # Only close if we created a new loop
            if loop != asyncio.get_event_loop():
                loop.close()    
    return wrapper

def get_all_commands() -> Dict[str, List[str]]:
    """Get all available commands grouped by provider"""
    commands = {}
    for provider_name, provider_class in ProviderRegistry._providers.items():
        plugin = ProviderPlugin(provider_class)
        commands[provider_name] = list(plugin.get_commands().keys())
    return commands

def print_command_help():
    """Print formatted help for all available commands"""
    table = Table(title="Available Commands")
    table.add_column("Provider", style="cyan")
    table.add_column("Command", style="green")
    table.add_column("Usage", style="yellow")
    
    commands = get_all_commands()
    for provider, cmd_list in sorted(commands.items()):
        for cmd in sorted(cmd_list):
            table.add_row(
                provider,
                cmd,
                f"sekvo {provider}.{cmd} [OPTIONS] PROMPT"
            )
    
    console.print(table)
    
    # Add piping examples
    console.print("\n[yellow]Piping Examples:[/yellow]")
    console.print("  echo 'tell me a joke' | sekvo anthropic.generate")
    console.print("  cat prompt.txt | sekvo openai.generate")

@click.group(invoke_without_command=True)
@click.option('--list-commands', is_flag=True, help="List all available commands")
@click.pass_context
def cli(ctx, list_commands):
    """sekvo CLI tool - A unified interface for various AI providers
    
    Examples:
    \b
    # Basic usage
    sekvo anthropic.generate "Tell me a joke"
    
    # With system prompt
    sekvo anthropic.generate -s "You are a poet" "Write a poem"
    
    # Piping
    echo "Tell me a joke" | sekvo anthropic.generate
    
    # Chaining providers
    echo "Write a story" | sekvo anthropic.generate | sekvo openai.generate
    """
    if list_commands or ctx.invoked_subcommand is None:
        print_command_help()

@cli.command(name="providers")
def list_providers():
    """List all available providers with their status"""
    table = Table(title="Available Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Commands", style="yellow")
    
    for provider_name, provider_class in sorted(ProviderRegistry._providers.items()):
        try:
            provider = provider_class()
            status = "✓ Ready"
            style = "green"
        except Exception as e:
            status = f"⚠ Not configured: {str(e)}"
            style = "red"
            
        commands = get_all_commands()[provider_name]
        table.add_row(
            provider_name,
            status,
            ", ".join(commands),
            style=style
        )
    
    console.print(table)

def load_provider_commands():
    """Load all provider commands into the CLI"""
    for provider_name, provider_class in ProviderRegistry._providers.items():
        plugin = ProviderPlugin(provider_class)
        for cmd_name, cmd in plugin.get_commands().items():
            # Wrap the callback with async_command decorator
            original_callback = cmd.callback
            cmd.callback = async_command(original_callback)
            
            # Add command help template
            if not cmd.help:
                cmd.help = f"""
                {cmd_name.replace('_', ' ').title()} using {provider_name.title()} provider

                Examples:
                \b
                # Basic usage
                sekvo {provider_name}.{cmd_name} "Your prompt here"
                
                # With system prompt
                sekvo {provider_name}.{cmd_name} -s "Custom system prompt" "Your prompt"
                
                # Using pipe
                echo "Your prompt" | sekvo {provider_name}.{cmd_name}
                """
            # Register command with format: provider_name.command_name
            cli.add_command(cmd, name=f"{provider_name}.{cmd_name}")

def main():
    """Entry point for the CLI"""
    try:
        load_provider_commands()
        cli()
    except Exception as e:
        console.print(Panel(
            f"[red]Error:[/red] {str(e)}",
            title="Error",
            border_style="red"
        ))

if __name__ == "__main__":
    main()