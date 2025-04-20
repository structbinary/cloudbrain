from typing import Dict, Any, List, Optional
from graph.state import CloudBrainState
from .base_agent import BaseAgent
from langchain.schema import Document
from graph.tools.factory import SearchFactory
import os


class SearcherAgent(BaseAgent):
    """Agent responsible for performing web searches to find relevant information."""

    def __init__(self, search_factory: SearchFactory, max_results=3, websocket=None, stream_output=None, headers=None):
        """Initialize the searcher agent.
        
        Args:
            search_factory: Factory for creating search instances
            max_results: Maximum number of search results to return
            websocket: WebSocket for streaming output
            stream_output: Function to stream output
            headers: Headers for API requests
        """
        super().__init__(websocket, stream_output, headers)
        self.max_results = max_results
        self.agent_name = "SEARCHER"
        self.search_factory = search_factory
    
    async def run(self, state: CloudBrainState) -> CloudBrainState:
        """
        Main entry point for the searcher agent.
        
        Args:
            state: The current state of the graph.
            
        Returns:
            The updated state with web search results.
        """
        return await self.search_web(state)
    
    async def search_web(self, state: CloudBrainState) -> CloudBrainState:
        """
        Performs a web search and adds the results to the state.
        
        Args:
            state: The current state of the graph.
            
        Returns:
            The updated state with web search results.
        """
        # Log the start of web search
        await self._log("Starting web search", self.agent_name)
        
        try:
            # Extract query and existing documents from state, handling both dictionary and object access
            if isinstance(state, dict):
                query = state.get("user_query")
                existing_documents = state.get("all_documents", [])
            else:
                query = state.user_query
                existing_documents = state.all_documents
                
            if not query:
                raise ValueError("No user query found in state")
            
            # Log the search query
            await self._log(f"Searching web for: {query}", self.agent_name)
            
            # Perform the web search
            search_results = await self._perform_web_search(query)
            
            # Process the search results
            web_documents = self._process_search_results(search_results)
            
            # Combine with existing documents
            all_documents = self._combine_documents(existing_documents, web_documents)
            
            # Log completion
            await self._log(f"Web search completed, found {len(web_documents)} new documents", self.agent_name)
            
            # Update and return the state
            return self._update_state(state, {
                "all_documents": all_documents,
                "web_documents": web_documents,
                "web_search_needed": False
            })
            
        except Exception as e:
            # Log error
            await self._log(f"Error during web search: {e}", self.agent_name)
            
            # Return the state unchanged
            return state
    
    async def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a web search using the search factory.
        
        Args:
            query: The search query
            
        Returns:
            List of search results
        """
        # Get API key from environment variable or configuration
        api_key = os.getenv("TAVILY_API_KEY")
        
        search_provider = self.search_factory.create_search(
            provider="tavily",
            api_key=api_key,
            max_results=self.max_results
        )
        return await search_provider.search(query)
    
    def _process_search_results(self, search_results: List[Dict[str, Any]]) -> List[Document]:
        """Process search results into Document objects.
        
        Args:
            search_results: List of search results
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for result in search_results:
            # Create a Document for each search result
            doc = Document(
                page_content=result["content"],
                metadata={"source": "web", "title": result.get("title", ""), "url": result.get("url", "")}
            )
            documents.append(doc)
        
        return documents
    
    def _combine_documents(self, existing_documents: List[Document], new_documents: List[Document]) -> List[Document]:
        """Combine existing documents with new documents.
        
        Args:
            existing_documents: List of existing documents
            new_documents: List of new documents
            
        Returns:
            Combined list of documents
        """
        # If there are no existing documents, return just the new ones
        if not existing_documents:
            return new_documents
        
        # Otherwise, combine them
        return existing_documents + new_documents

# For backward compatibility with AgentRunner
def websearch(search_factory: SearchFactory, websocket=None, stream_output=None, headers=None, max_results=3):
    """
    Factory function to create a SearcherAgent instance.
    
    Args:
        search_factory: Factory for creating search instances
        websocket: WebSocket for streaming output
        stream_output: Function to stream output
        headers: Headers for API requests
        max_results: Maximum number of search results to return
        
    Returns:
        A function that takes a state and returns the updated state
    """
    agent = SearcherAgent(search_factory, max_results, websocket, stream_output, headers)
    
    async def run(state: CloudBrainState) -> CloudBrainState:
        """
        Run the searcher agent.
        
        Args:
            state: The current state of the graph.
            
        Returns:
            The updated state with web search results.
        """
        return await agent.search_web(state)
    
    return run
