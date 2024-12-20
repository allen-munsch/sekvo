import asyncio
from sekvo.providers.anthropic.generate import AnthropicProvider


async def generate_text():
    prompt = 'tell me a joke'
    system_prompt = 'you tell jokes in rhymes'
    provider = AnthropicProvider(env_name="anthropic-dev")
    response = await provider.generate(prompt, system_prompt)
    print(response)



asyncio.run(generate_text())