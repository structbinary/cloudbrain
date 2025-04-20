import asyncio
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from cloudbrain.graph.llm.factory import LLMFactory

# Load environment variables
load_dotenv()

class SimpleWebSocket:
    """A simple WebSocket mock for demonstration purposes."""
    
    async def send_json(self, data):
        """Send JSON data to the client."""
        if data["type"] == "report":
            print(data["output"], end="", flush=True)

async def main():
    """Run an example of streaming with the LLM package."""
    # Create a simple WebSocket mock
    websocket = SimpleWebSocket()
    
    # Create an LLM instance with streaming enabled
    llm = LLMFactory.create_llm(
        provider="openai",  # or "anthropic" for Claude
        model="gpt-3.5-turbo",  # or "claude-3-opus-20240229" for Claude
        temperature=0.7,
        max_tokens=500,
        stream=True
    )
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Write a short story about a developer who discovers a magical programming language.")
    ]
    
    # Generate text with streaming
    print("Generating response (streaming)...\n")
    response = await llm.generate(messages, websocket=websocket)
    
    # Print a newline at the end
    print("\n\nStreaming complete!")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 