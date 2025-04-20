"""
Factory for creating search instances.
"""

from typing import Any, Dict, Optional
from .base import GenericSearchProvider

class SearchFactory:
    """Factory for creating search instances."""
    
    @staticmethod
    def create_search(
        provider: str,
        api_key: Optional[str] = None,
        **kwargs: Any
    ) -> GenericSearchProvider:
        """Create a search instance.
        
        Args:
            provider: The search provider to use
            api_key: API key for the provider
            **kwargs: Additional arguments for the provider
            
        Returns:
            A search instance
        """
        return GenericSearchProvider.from_provider(
            provider=provider,
            api_key=api_key,
            **kwargs
        ) 