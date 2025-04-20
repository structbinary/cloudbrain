"""
Provider for LangChain's ChatOpenAI.
"""

from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage
from ..base import GenericLLMProvider

class OpenAIProvider(GenericLLMProvider):
    """Provider for LangChain's ChatOpenAI."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ):
        """Initialize the LangChain OpenAI provider.
        
        Args:
            model: The OpenAI model to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments for the ChatOpenAI model
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        
        # Initialize the ChatOpenAI model
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=stream,
            **kwargs
        )

    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> str:
        """Generate text using LangChain's ChatOpenAI.
        
        Args:
            messages: List of messages to generate from
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text
        """
        response = await self.llm.ainvoke(messages)
        return response.content 