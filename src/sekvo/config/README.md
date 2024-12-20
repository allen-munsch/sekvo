# The loading order is:

1) Base .env file
2) Provider base file (e.g., .env.anthropic)
3) Provider-environment specific file (e.g., .env.anthropic.dev)
4) Shell environment variables (these override everything)


# Initialize provider with specific environment

```
provider = AnthropicProvider(env_name="anthropic-dev")
```

# Or use settings directly

```
settings = SekvoSettings.from_env("anthropic-dev")
print(settings.anthropic.api_key)  # dev-key
print(settings.openai)  # None (because we specified anthropic environment)
```

# Use in async context

```
async def generate_text(prompt: str):
    provider = AnthropicProvider(env_name="anthropic-dev")
    return await provider.generate(prompt)
```

# Shell environment export:

```
# Export directly to shell
export SEKVO_ANTHROPIC_API_KEY=my-key
export SEKVO_ANTHROPIC_MODEL=claude-3-opus-20240229
export SEKVO_ANTHROPIC_ADDITIONAL_PARAMS_TEMPERATURE=0.8
```