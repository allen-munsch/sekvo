![Sekvo logo](./docs/sekvo-logo.jpg)

# sekvo

Sekvo is a simple and flexible Python library for piping and processing of prompts through various AI providers. 

With Sekvo, you can easily chain prompts, use multiple providers, collect metrics, validate responses, apply transformations and filters, and process prompts in parallel and batches, all from the comfort of a shell.

![Python Version](https://img.shields.io/badge/python-%3E%3D%203.13-blue)
# quick usage and installation example

```
# via git
cd sekvo/
pyenv install 3.13.1
python -m venv venv_3131/
pip isntall -e '.[all]'

# via pypi
pip install sekvo


# add your key to .env/.env.anthropic.dev

# usage
sekvo
sekvo --help
sekvo providers
sekvo --list-commands
sekvo anthropic.generate --help

echo 'tell me a joke' | sekvo anthropic.generate
sekvo anthropic.generate 'tell me a joke'

# raw output
echo 'tell me a joke' | sekvo anthropic.generate --raw

# json output
sekvo anthropic.generate --json 'tell me a joke'

# a custom prompt and summary
curl https://nebkiso.com > page.txt
echo 'What is this webpage about?\n' > prompt.txt
cat prompt.txt page.txt | sekvo anthropic.generate


# structured json merging

jq -s 'add | unique' <(cat my-json-prompt.txt | sekvo anthropic.generate --raw) ../old-places.json >| merged-places.json

```

## Features

- **Prompt Piping**: Chain prompts together and pipe them through different AI providers for sequential processing.
- **Multiple Providers**: Seamlessly integrate with multiple AI providers, such as AnthropicProvider, and switch between them effortlessly.
- **Parallel Processing**: Process prompts in parallel using multiple providers simultaneously.
- **Metrics Collection**: Collect valuable metrics, including provider information, input tokens, and processing duration, for each prompt.
- **JSON Validation**: Automatically validate the JSON response from providers and retry on invalid responses.
- **Transformations and Filters**: Apply custom transformations and filters to the prompts and responses.
- **Batch Processing**: Process multiple prompts in batches for efficient handling of large datasets.

## More Usage Examples

## Examples

More examples: [Located here](docs/)

### Basic Prompt Processing

```python
from sekvo.core.prompt_pipe import Prompt
from sekvo.providers.anthropic.generate import AnthropicProvider

provider = AnthropicProvider(env_name="anthropic-test")

result = await (Prompt("test prompt") | provider)
print(result)  # Output: response(test prompt)
```

### Chaining Prompts and Providers

```python
from sekvo.core.prompt_pipe import Prompt
from sekvo.providers.anthropic.generate import AnthropicProvider

provider1 = AnthropicProvider(env_name="anthropic-test1")
provider2 = AnthropicProvider(env_name="anthropic-test2")

result = await (Prompt("test prompt") | provider1 | provider2)
print(result)  # Output: provider2(provider1(test prompt))
```

### Parallel Processing

```python
from sekvo.core.prompt_pipe import ParallelPrompt
from sekvo.providers.anthropic.generate import AnthropicProvider

provider1 = AnthropicProvider(env_name="anthropic-test1")
provider2 = AnthropicProvider(env_name="anthropic-test2")

result = await (ParallelPrompt("test prompt") | [provider1, provider2])
print(result)  # Output: ["provider1(test prompt)", "provider2(test prompt)"]
```

### Transformations and Filters

```python
from sekvo.core.prompt_pipe import FilterPrompt
from sekvo.providers.anthropic.generate import AnthropicProvider

provider = AnthropicProvider(env_name="anthropic-test")

prompt = FilterPrompt("test prompt")
result = await (
    prompt
    .transform(str.upper)
    .filter(lambda x: len(x) > 0) |
    provider
)
print(result)  # Output: response(TEST PROMPT)
```

### Batch Processing

```python
from sekvo.core.prompt_pipe import BatchPrompt
from sekvo.providers.anthropic.generate import AnthropicProvider

provider = AnthropicProvider(env_name="anthropic-test")

prompt = BatchPrompt(["test1", "test2", "test3"])
results = await (prompt | provider)
print(results)  # Output: ["response(test1)", "response(test2)", "response(test3)"]
```

## Installation

```bash
pip install sekvo
```

## Contributing

Contributions are welcome!
