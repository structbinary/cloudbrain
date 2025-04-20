"""
Example of using the Tavily search provider.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(parent_dir)

from dotenv import load_dotenv
from cloudbrain.graph.tools.factory import SearchFactory

# Load environment variables
load_dotenv()

async def main():
    """Run an example of using the Tavily search provider."""
    # Create a search instance
    search = SearchFactory.create_search(
        provider="tavily",
        api_key=os.getenv("TAVILY_API_KEY")
    )
    
    # Perform a search
    print("Performing search...")
    results = await search.search(
        query="What are the benefits of using Terraform for infrastructure as code?",
        max_results=5
    )
    
    # Print the results
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {result.get('title')}")
        print(f"URL: {result.get('url')}")
        print(f"Content: {result.get('content')[:200]}...")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 