"""
Example of using the LangChain OpenAI provider with templates.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(parent_dir)

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from cloudbrain.graph.llm.factory import LLMFactory

# Load environment variables
load_dotenv()

async def main():
    """Run an example of using the LangChain OpenAI provider with templates."""
    # Create an LLM instance
    llm = LLMFactory.create_llm(
        provider="langchain_openai",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
        stream=False
    )
    
    # Create a prompt template
    template = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Answer the following question:

Question: {question}

Answer:"""
    )
    
    # Format the template with variables
    formatted_messages = template.format_messages(
        question="What are the benefits of using Terraform for infrastructure as code?"
    )
    
    # Generate text using the formatted messages
    print("Generating response...")
    response = await llm.generate(formatted_messages)
    
    # Print the response
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 