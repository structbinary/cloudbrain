"""
Factory for creating LLM instances.
"""

from typing import Any, Dict, Optional
from .base import BaseLLM
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field


class LangChainCompatibleLLM(BaseChatModel):
    """A wrapper around BaseLLM that implements the LangChain Runnable interface."""
    
    base_llm: BaseLLM = Field(description="The underlying BaseLLM instance")
    model_name: str = Field(description="The name of the model")
    temperature: float = Field(default=0.7, description="The temperature to use for generation")
    max_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens to generate")
    
    def __init__(self, base_llm: BaseLLM, **kwargs):
        """Initialize with a BaseLLM instance.
        
        Args:
            base_llm: The underlying BaseLLM instance
            **kwargs: Additional arguments for the parent class
        """
        # Extract values from base_llm
        model_name = base_llm.model
        temperature = base_llm.temperature
        max_tokens = base_llm.max_tokens
        streaming = base_llm.stream  # Use parent's streaming field
        
        # Initialize the parent class with the extracted values
        super().__init__(
            base_llm=base_llm,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,  # Use parent's streaming field
            **kwargs
        )
    
    def _generate(
        self, messages: list[BaseMessage], stop: Optional[list[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any
    ) -> ChatResult:
        """Generate a chat result from the messages (synchronous version).
        
        Args:
            messages: List of messages to generate from
            stop: Optional list of stop sequences
            run_manager: Optional run manager
            **kwargs: Additional arguments for generation
            
        Returns:
            A ChatResult object
        """
        # This is a synchronous wrapper around the async method
        import asyncio
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(self.base_llm.generate(messages, **kwargs))
        
        # Create a simple ChatGeneration with the response text
        generation = ChatGeneration(
            text=response,
            message=BaseMessage(content=response, type="ai")
        )
        
        return ChatResult(generations=[generation])
    
    async def _agenerate(
        self, messages: list[BaseMessage], stop: Optional[list[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any
    ) -> ChatResult:
        """Generate a chat result from the messages (asynchronous version).
        
        Args:
            messages: List of messages to generate from
            stop: Optional list of stop sequences
            run_manager: Optional run manager
            **kwargs: Additional arguments for generation
            
        Returns:
            A ChatResult object
        """
        response = await self.base_llm.generate(messages, **kwargs)
        
        # Create a simple ChatGeneration with the response text
        generation = ChatGeneration(
            text=response,
            message=BaseMessage(content=response, type="ai")
        )
        
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type.
        
        Returns:
            The LLM type
        """
        return f"custom-{self.base_llm.provider}"


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Runnable:
        """Create an LLM instance.
        
        Args:
            provider: The LLM provider to use
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments for the provider
            
        Returns:
            A LangChain-compatible LLM instance
        """
        class DefaultLLM(BaseLLM):
            async def generate(self, messages, **kwargs):
                return await self._provider.generate(messages, **kwargs)
        
        base_llm = DefaultLLM(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
        
        # Wrap the BaseLLM in a LangChain-compatible interface
        return LangChainCompatibleLLM(base_llm)
    
    @staticmethod
    def create_embeddings(
        provider: str,
        model: str,
        **kwargs: Any
    ) -> Any:
        """Create an embeddings instance.
        
        Args:
            provider: The embeddings provider to use
            model: The model to use
            **kwargs: Additional arguments for the provider
            
        Returns:
            An embeddings instance
        """
        # Extract vector store specific parameters
        vector_store_params = {
            'persist_directory': kwargs.pop('persist_directory', None),
            'collection_name': kwargs.pop('collection_name', None)
        }
        
        # Create embeddings with the remaining parameters
        if provider.lower() == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=model,
                **kwargs
            )
        elif provider.lower() == "anthropic":
            from langchain_anthropic import AnthropicEmbeddings
            return AnthropicEmbeddings(
                model=model,
                **kwargs
            )
        elif provider.lower() == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=model,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported embeddings provider: {provider}") 