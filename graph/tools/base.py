"""
Base module for search implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class GenericSearchProvider:
    """Generic provider class for search API calls."""

    @staticmethod
    def from_provider(
        provider: str,
        api_key: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """Create a provider instance based on the specified provider.
        
        Args:
            provider: The search provider to use
            api_key: API key for the provider
            **kwargs: Additional arguments for the provider
            
        Returns:
            A provider instance
            
        Raises:
            ValueError: If the provider is not supported
        """
        if provider.lower() == "tavily":
            from .providers.tavily_provider import TavilyProvider
            return TavilyProvider(
                api_key=api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def search(
        self,
        query: str,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Perform a search query.
        
        Args:
            query: The search query
            **kwargs: Additional arguments for the search
            
        Returns:
            List of search results
        """
        raise NotImplementedError("Subclasses must implement search()") 