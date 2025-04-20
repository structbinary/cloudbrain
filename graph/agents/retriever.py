from typing import Dict, Any, List, Optional
from graph.state import CloudBrainState
from .base_agent import BaseAgent
from langchain_chroma import Chroma
from langchain.schema import Document

"""Need to fix the Chroma vector store. This should go in the config file. and from the workflow manager the vector store should be passed in as a parameter"""
class RetrieverAgent(BaseAgent):
    """Agent responsible for retrieving relevant documents from a vector store."""

    def __init__(self, embeddings=None, config=None, websocket=None, stream_output=None, headers=None):
        """Initialize the retriever agent.
        
        Args:
            embeddings: Optional embeddings instance to use for vector search
            config: Configuration dictionary for vector store settings
            websocket: WebSocket for streaming output
            stream_output: Function to stream output
            headers: Headers for API requests
        """
        super().__init__(websocket, stream_output, headers)
        self.config = config or {}
        self.agent_name = "RETRIEVER"
        self.min_documents_threshold = 2  # Minimum number of documents to consider search successful
        
        # Debug: Print config
        # print(f"DEBUG: Config received: {self.config}")
        
        # Extract vector store specific config
        self.vector_store_config = {
            'persist_directory': self.config.get('persist_directory', "./vectorstore")
        }
        
        # Use the embeddings passed from the workflow manager
        self.embeddings = embeddings
        # print(f"DEBUG: Using embeddings: {embeddings}")
    
    async def run(self, state: CloudBrainState) -> CloudBrainState:
        """
        Main entry point for the retriever agent.
        
        Args:
            state: The current state of the graph.
            
        Returns:
            The updated state with retrieved documents.
        """
        # Log retrieval start
        await self._log("Starting document retrieval", self.agent_name)
        
        try:
            # Get documents from state if available, handling both dictionary and object access
            if isinstance(state, dict):
                documents = state.get("documents", [])
                query = state.get("user_query")
            else:
                documents = state.documents if hasattr(state, 'documents') else []
                query = state.user_query
                
            await self._log(f"Documents from state: {len(documents) if documents else 0}", self.agent_name)
            
            # Use the same configuration as defined in config.py
            persist_directory = self.config.get('persist_directory', './data/chroma')
            collection_name = self.config.get('collection_name', 'terraform_docs')
            await self._log(f"Using vector store at: {persist_directory} with collection: {collection_name}", self.agent_name)
            
            # Initialize vector store with documents if available
            vector_store = self._initialize_vector_store(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                documents=documents
            )
            
            # Get query from state
            if not query:
                raise ValueError("No user query found in state")
            
            # Debug: Check if vector store has any documents
            collection = vector_store._collection
            count = collection.count()
            await self._log(f"Vector store contains {count} documents", self.agent_name)
            
            # Perform similarity search
            await self._log(f"Searching for documents matching query: {query}", self.agent_name)
            results = vector_store.similarity_search(query, k=self.min_documents_threshold)
            
            # Determine if web search is needed
            web_search_needed = len(results) < self.min_documents_threshold
            
            # Log retrieval results
            await self._log(f"Retrieved {len(results)} documents", self.agent_name)
            if web_search_needed:
                await self._log("Not enough documents found, web search will be needed", self.agent_name)
            
            # Update and return the state
            return self._update_state(state, {
                "retrieved_documents": results,
                "all_documents": results,  # Initially, all documents are local
                "web_search_needed": web_search_needed
            })
            
        except Exception as e:
            # Log error
            await self._log(f"Error searching vector store: {e}", self.agent_name)
            
            # If there's an error, we'll need to do a web search
            return self._update_state(state, {
                "retrieved_documents": [],
                "all_documents": [],
                "web_search_needed": True
            })
    
    def _initialize_vector_store(self, persist_directory: str, embedding_function, documents: Optional[List[Document]] = None) -> Chroma:
        """Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector store
            embedding_function: Embedding function to use
            documents: Optional list of documents to initialize the vector store with
            
        Returns:
            Chroma: Initialized vector store
        """
        # Debug: Print parameters
        # print(f"DEBUG: Initializing vector store with persist_directory={persist_directory}")
        # print(f"DEBUG: Documents provided: {len(documents) if documents else 0}")
        
        # Use the same collection name as defined in config.py
        collection_name = self.config.get('collection_name', 'terraform_docs')
        # print(f"DEBUG: Using collection name: {collection_name}")
        
        if documents:
            # Initialize Chroma vector store with documents
            # print(f"DEBUG: Creating vector store with {len(documents)} documents")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        else:
            # Initialize empty Chroma vector store
            # print(f"DEBUG: Creating empty vector store")
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_function,
                collection_name=collection_name
            )
        
        return vector_store


def retrieve(websocket=None, stream_output=None, headers=None, embeddings=None, config=None):
    """Factory function to create a RetrieverAgent instance.
    
    Args:
        websocket: WebSocket for streaming output
        stream_output: Function to stream output
        headers: Headers for API requests
        embeddings: Optional embeddings instance to use for vector search
        config: Configuration dictionary for vector store settings
        
    Returns:
        A function that takes a state and returns an updated state
    """
    agent = RetrieverAgent(embeddings, config, websocket, stream_output, headers)
    
    async def run(state: CloudBrainState) -> CloudBrainState:
        """Run the retriever agent.
        
        Args:
            state: The current state of the graph.
            
        Returns:
            The updated state with retrieved documents.
        """
        return await agent.run(state)
    
    return run