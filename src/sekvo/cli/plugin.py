import json as jsonlib
import sys

import click
from rich.console import Console
from rich.panel import Panel

from sekvo.config.settings import ENV_NAME
from sekvo.providers import ProviderRegistry
from sekvo.providers.base import BaseProvider

console = Console()

class PipedCommand(click.Command):
    def invoke(self, ctx):
        # Check for piped input
        if not sys.stdin.isatty():
            ctx.params['prompt'] = sys.stdin.read().strip()
        return super().invoke(ctx)


class ProviderPlugin:
    def __init__(self, provider_class: type[BaseProvider]):
        self.provider_class = provider_class

    @property
    def name(self) -> str:
        for name, provider in ProviderRegistry._providers.items():
            if provider == self.provider_class:
                return name
        raise ValueError("Provider not registered")

    def get_commands(self) -> dict[str, click.Command]:
        @click.command(cls=PipedCommand)
        @click.argument('prompt', required=False)
        @click.option("--system-prompt", "-s",
                    default="You are a helpful assistant.",
                    help="System prompt to use")
        @click.option("--env", default=ENV_NAME,
                    help="Environment name for config")
        @click.option("--raw", "-r", is_flag=True,
                    help="Output raw text without formatting")
        @click.option("--json", "-j", is_flag=True,
                    help="Output generation as json")
        @click.pass_context
        async def generate(ctx, prompt: str, system_prompt: str, env: str, raw: bool, json: bool):
            """Generate text using the provider"""
            try:
                provider = self.provider_class(env_name=env)
                result = await provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt
                )

                if raw:
                    click.echo(result)
                elif json:
                    click.echo(jsonlib.dumps({"value": result}))
                else:
                    console.print(Panel(
                        result,
                        title=f"{self.name.capitalize()} Generation",
                        title_align="left"
                    ))
            except Exception as e:
                if raw:
                    click.echo(f"Error: {e!s}", err=True)
                else:
                    console.print(f"[red]Error:[/red] {e!s}")
                sys.exit(1)

        @click.command(cls=PipedCommand)
        @click.argument('prompt', required=False)
        @click.option("--system-prompt", "-s",
                    default="You are a helpful assistant.",
                    help="System prompt to use")
        @click.option("--env", default=ENV_NAME,
                    help="Environment name for config")
        @click.option("--raw", "-r", is_flag=True,
                    help="Output raw text without formatting")
        @click.pass_context
        async def stream(ctx, prompt: str, system_prompt: str, env: str, raw: bool):
            """Stream text generation using the provider"""
            from rich.live import Live
            from rich.text import Text
            try:
                provider = self.provider_class(env_name=env)

                if not provider.supports_streaming:
                    console.print(f"[yellow]Warning:[/yellow] {self.name.capitalize()} does not support streaming. Falling back to regular generation.")

                if raw:
                    # For raw output, just print tokens as they arrive
                    async for token in provider.generate_stream(prompt, system_prompt):
                        click.echo(token, nl=False)
                        sys.stdout.flush()  # Ensure tokens are displayed immediately
                    click.echo()  # Final newline
                else:
                    # For rich output, use Live display with continuous updates
                    text = Text()
                    with Live(Panel(text, title=f"{self.name.capitalize()} Generation (Streaming)"), refresh_per_second=15) as live:
                        async for token in provider.generate_stream(prompt, system_prompt):
                            text.append(token)
                            live.update(Panel(text, title=f"{self.name.capitalize()} Generation (Streaming)"))

            except Exception as e:
                if raw:
                    click.echo(f"Error: {e!s}", err=True)
                else:
                    console.print(f"[red]Error:[/red] {e!s}")
                sys.exit(1)
        # Return the dictionary of commands
        return {"generate": generate, "stream": stream}
