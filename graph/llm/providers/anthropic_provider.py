from typing import Any, Dict, List, Optional
import anthropic
from langchain.schema import BaseMessage
from ..base import GenericLLMProvider

class AnthropicProvider(GenericLLMProvider):
    """Provider for Anthropic's API."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any
    ):
        """Initialize the Anthropic provider.
        
        Args:
            model: The Anthropic model to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments for the Anthropic client
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.client = anthropic.Anthropic(**kwargs)

    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> str:
        """Generate text using Anthropic's API.
        
        Args:
            messages: List of messages to generate from
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text
        """
        response = await self.client.messages.create(
            model=self.model,
            messages=[{"role": msg.type, "content": msg.content} for msg in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=self.stream,
            **kwargs
        )
        
        if self.stream:
            # Handle streaming response
            collected_messages = []
            async for chunk in response:
                if chunk.delta.text:
                    collected_messages.append(chunk.delta.text)
            return "".join(collected_messages)
        else:
            # Handle non-streaming response
            return response.content[0].text 