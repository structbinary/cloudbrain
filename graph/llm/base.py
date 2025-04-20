"""
Base module for LLM implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain.schema import BaseMessage

class GenericLLMProvider:
    """Generic provider class for LLM API calls."""

    @staticmethod
    def from_provider(
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Any:
        """Create a provider instance based on the specified provider.
        
        Args:
            provider: The LLM provider to use
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments for the provider
            
        Returns:
            A provider instance
            
        Raises:
            ValueError: If the provider is not supported
        """
        if provider.lower() == "openai":
            from .providers.openai_provider import OpenAIProvider
            return OpenAIProvider(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
        elif provider.lower() == "anthropic":
            from .providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
        # elif provider.lower() == "langchain_openai":
        #     from .providers.langchain_openai_provider import LangChainOpenAIProvider
        #     return LangChainOpenAIProvider(
        #         model=model,
        #         temperature=temperature,
        #         max_tokens=max_tokens,
        #         stream=stream,
        #         **kwargs
        #     )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> str:
        """Generate text from the LLM.
        
        Args:
            messages: List of messages to generate from
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text
        """
        raise NotImplementedError("Subclasses must implement generate()")

class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ):
        """Initialize the LLM.
        
        Args:
            provider: The LLM provider to use
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments for the provider
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.kwargs = kwargs
        
        # Create the provider instance
        self._provider = GenericLLMProvider.from_provider(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
    
    @abstractmethod
    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> str:
        """Generate text from the LLM.
        
        Args:
            messages: List of messages to generate from
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text
        """
        pass 