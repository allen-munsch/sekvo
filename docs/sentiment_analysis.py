import asyncio
from sekvo.core.prompt_pipe import ParallelPrompt, FilterPrompt, BatchPrompt
from sekvo.providers.anthropic.generate import AnthropicProvider

# Initialize the Anthropic providers with example environment names (adjust your env config as needed)
provider1 = AnthropicProvider(env_name="anthropic-dev")
provider2 = AnthropicProvider(env_name="anthropic-dev")

# Sample customer feedback
feedbacks = [
    "I love the service, it was very quick and helpful. Will use again!",
    "The product is terrible. It broke after one use and customer support didn't help.",
    "Amazing experience! The team was friendly and resolved my issue in no time.",
    "I'm dissatisfied with the delivery. It was late and the packaging was damaged.",
    "Great! Everything was smooth, but I think the pricing could be better."
]

# Prompt 1: Identifying negative sentiment
def negative_sentiment_prompt(feedback):
    return f"Is this feedback negative? Please respond with 'yes' or 'no'.\n\nFeedback: {feedback}"

# Prompt 2: Extracting metadata (sentiment strength and keywords)
def metadata_prompt(feedback):
    return f"Please provide the sentiment strength (positive, negative, neutral) and extract any relevant keywords or context from this feedback:\n\nFeedback: {feedback}"

async def sentiment_analysis():
    # Process the feedback in parallel through two different prompts
    prompts = ParallelPrompt(feedbacks)

    # Define the parallel processing where each feedback is analyzed by both prompts simultaneously
    result = await (
        prompts
        .transform(lambda feedback: (negative_sentiment_prompt(feedback), metadata_prompt(feedback)))  # Transform to two prompts
        | [provider1, provider2]  # Send both prompts to two providers for parallel processing
    )

    # Process and surface only negative feedback along with metadata
    print("Negative Feedback with Sentiment Analysis and Metadata:")
    for feedback, (sentiment_result, metadata_result) in zip(feedbacks, result):
        sentiment = sentiment_result.strip().lower()  # Normalize the sentiment result
        if sentiment == "yes":  # If the feedback is negative
            print(f"\nFeedback: {feedback}")
            print(f"Sentiment: Negative")
            print(f"Metadata: {metadata_result}")

async def summarize_feedback():
    # Summarizing feedback and applying filters to remove empty or irrelevant responses
    prompt = FilterPrompt(feedbacks)
    
    # Apply transformation to summarize feedback, and filter out incomplete responses
    result = await (
        prompt
        .transform(lambda x: f"Summarized: {x[:50]}...")  # Only the first 50 characters as a summary
        .filter(lambda x: len(x) > 20)  # Filter out feedback that's too short to be relevant
        | provider1  # Only send to one provider for now
    )

    print("\nSummarized and filtered feedback:")
    for res in result:
        print(res)

async def batch_sentiment_analysis():
    # Batch processing for feedback analysis
    prompt = BatchPrompt(feedbacks)
    
    # Process each feedback in a batch with the first provider
    results = await (prompt | provider1)
    
    print("\nBatch processing results:")
    for feedback, res in zip(feedbacks, results):
        print(f"Feedback: {feedback}\nSentiment: {res}\n")

async def main():
    # Run all the tasks asynchronously
    await asyncio.gather(
        sentiment_analysis(),
        summarize_feedback(),
        batch_sentiment_analysis(),
    )

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
