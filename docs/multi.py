import asyncio
from pydantic import BaseModel, Field
from typing import List

from sekvo.core.prompt_pipe import Prompt, ParallelPrompt, MetricsPrompt, ValidatedPrompt
from sekvo.providers.simplemind_adapter import (
    AnthropicProvider, 
    OpenAIProvider, 
    GroqProvider,
    GeminiProvider,
    OllamaProvider
)

# Define a structured response model
class MovieRecommendation(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    reason: str = Field(description="Why this movie is recommended")

async def basic_example():
    """Basic example using a single provider"""
    provider = AnthropicProvider(env_name="anthropic-dev")
    prompt = Prompt("Recommend a sci-fi movie from the 1980s", 
                   system_prompt="You are a helpful movie recommendation assistant.")
    
    result = await (prompt | provider)
    print(f"Basic Result:\n{result}\n")

async def parallel_example():
    """Compare responses from multiple providers"""
    anthropic = AnthropicProvider(env_name="anthropic-dev")
    openai = OpenAIProvider(env_name="openai-dev")
    groq = GroqProvider(env_name="groq-dev")
    
    prompt = ParallelPrompt("What's the most interesting application of quantum computing?")
    
    # Process the same prompt with different providers in parallel
    results = await (prompt | [anthropic, openai, groq])
    
    print("Parallel Results:")
    for i, result in enumerate(results):
        provider_name = [anthropic, openai, groq][i].__class__.__name__
        print(f"\n--- {provider_name} ---\n{result}")

async def metrics_example():
    """Collect performance metrics while generating content"""
    provider = AnthropicProvider(env_name="anthropic-dev")
    prompt = MetricsPrompt("Explain how neural networks work in simple terms")
    
    result = await (prompt | provider)
    
    print(f"Metrics Result:\n{result}\n")
    print(f"Performance Metrics:")
    for metric in prompt.metrics:
        print(f"Provider: {metric.provider}")
        print(f"Input tokens (approx): {metric.input_tokens}")
        print(f"Output tokens (approx): {metric.output_tokens}")
        print(f"Duration: {metric.duration:.2f} seconds")

async def structured_response_example():
    """Get a structured response from a provider"""
    provider = AnthropicProvider(env_name="anthropic-dev")
    
    # For structured responses, we use the provider's method directly
    movie = provider.structured_response(
        prompt="Recommend a sci-fi movie from the 1980s and format the response as requested",
        response_model=MovieRecommendation
    )
    
    print(f"Structured Response Example:")
    print(f"Title: {movie.title} ({movie.year})")
    print(f"Director: {movie.director}")
    print(f"Reason: {movie.reason}")

async def tool_example():
    """Use a provider with a tool"""
    provider = OpenAIProvider(env_name="openai-dev")
    
    # Define a simple calculator tool
    def calculator(expression: str = Field(description="A mathematical expression to evaluate")) -> str:
        """Calculate the result of a mathematical expression"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create a conversation to use the tool
    from simplemind.models import Conversation
    
    # Create a conversation
    conversation = Conversation()
    conversation.add_message(role="user", text="What is 1234 * 5678?")
    
    # Send the conversation with the tool
    tools = [calculator]
    response = provider.simplemind_provider.send_conversation(conversation, tools=tools)
    
    print(f"Tool Example:")
    print(f"Question: What is 1234 * 5678?")
    print(f"Answer: {response.text}")

async def main():
    """Run all examples"""
    await basic_example()
    print("\n" + "="*50 + "\n")
    
    await parallel_example()
    print("\n" + "="*50 + "\n")
    
    await metrics_example()
    print("\n" + "="*50 + "\n")
    
    await structured_response_example()
    print("\n" + "="*50 + "\n")
    
    await tool_example()

if __name__ == "__main__":
    asyncio.run(main())