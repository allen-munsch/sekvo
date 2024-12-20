from unittest.mock import patch
from sekvo.providers.anthropic.generate import AnthropicProvider
import pytest
from sekvo.cli.main import cli


def test_direct_cli_usage(cli_runner, mock_env):
    """Test direct CLI usage with arguments"""
    result = cli_runner.invoke(cli, ['anthropic.generate', 'tell me a joke'])
    result2 = cli_runner.invoke(cli, ['--list-commands'])
    assert result.exit_code == 0, result.output
    assert "This is a test response" in result.output
    assert result2.exit_code == 0, result2.output
    assert "anthropic" in result2.output

def test_cli_with_options(cli_runner, mock_env):
    """Test CLI with various options"""
    result = cli_runner.invoke(cli, [
        'anthropic.generate',
        '--system-prompt', 'You are a comedian',
        '--raw',
        'tell me a joke'
    ])
    assert result.exit_code == 0
    assert result.output.strip() == "This is a test response"

def test_cli_pipe_input(cli_runner, mock_env):
    """Test CLI with piped input"""
    result = cli_runner.invoke(cli, ['anthropic.generate'], input='tell me a joke')
    assert result.exit_code == 0
    assert "This is a test response" in result.output

def test_cli_help(cli_runner):
    """Test CLI help command"""
    result = cli_runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output

def test_cli_provider_list(cli_runner):
    """Test provider listing command"""
    result = cli_runner.invoke(cli, ['providers'])
    assert result.exit_code == 0
    assert "anthropic" in result.output.lower()
