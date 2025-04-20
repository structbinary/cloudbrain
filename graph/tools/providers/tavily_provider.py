"""
Provider for Tavily's search API using LangChain's TavilySearchResults.
"""

from typing import Any, Dict, List, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from ..base import GenericSearchProvider

class TavilyProvider(GenericSearchProvider):
    """Provider for Tavily's search API using LangChain's TavilySearchResults."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 3,
        **kwargs: Any
    ):
        """Initialize the Tavily provider.
        
        Args:
            api_key: Tavily API key
            max_results: Maximum number of results to return
            **kwargs: Additional arguments for the TavilySearchResults
        """
        self.api_key = api_key
        self.max_results = max_results
        self.search_tool = TavilySearchResults(max_results=max_results, **kwargs)
    
    @classmethod
    def create_search(
        cls,
        provider: str,
        api_key: Optional[str] = None,
        **kwargs: Any
    ) -> "TavilyProvider":
        """Create a search instance.
        
        Args:
            provider: The search provider to use (ignored, always returns TavilyProvider)
            api_key: API key for the provider
            **kwargs: Additional arguments for the provider
            
        Returns:
            A TavilyProvider instance
        """
        return cls(api_key=api_key, **kwargs)

    async def search(
        self,
        query: str,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Perform a search using Tavily's API via LangChain's TavilySearchResults.
        
        Args:
            query: The search query
            **kwargs: Additional arguments for the search
            
        Returns:
            List of search results
        """
        # Use the invoke method of TavilySearchResults
        results = self.search_tool.invoke({"query": query})
        return results 