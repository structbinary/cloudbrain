import asyncio
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from cloudbrain.graph.llm.factory import LLMFactory

# Load environment variables
load_dotenv()

async def generate_with_provider(provider, model, prompt):
    """Generate text with a specific provider.
    
    Args:
        provider: The LLM provider to use
        model: The model to use
        prompt: The prompt to generate from
        
    Returns:
        Generated text
    """
    # Create an LLM instance
    llm = LLMFactory.create_llm(
        provider=provider,
        model=model,
        temperature=0.7,
        max_tokens=200,
        stream=False
    )
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]
    
    # Generate text
    print(f"Generating response with {provider} ({model})...")
    response = await llm.generate(messages)
    
    return response

async def main():
    """Run an example of using multiple providers with the LLM package."""
    # Define the prompt
    prompt = "What are the key differences between Python and JavaScript?"
    
    # Generate with OpenAI
    openai_response = await generate_with_provider(
        provider="openai",
        model="gpt-3.5-turbo",
        prompt=prompt
    )
    
    # Generate with Anthropic (if API key is available)
    anthropic_response = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        anthropic_response = await generate_with_provider(
            provider="anthropic",
            model="claude-3-opus-20240229",
            prompt=prompt
        )
    
    # Print the responses
    print("\n=== OpenAI Response ===")
    print(openai_response)
    
    if anthropic_response:
        print("\n=== Anthropic Response ===")
        print(anthropic_response)
    else:
        print("\n=== Anthropic Response ===")
        print("Anthropic API key not available. Skipping Anthropic generation.")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 