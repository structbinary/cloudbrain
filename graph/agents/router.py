from typing import Dict, Any, List, Optional, Literal
from graph.state import CloudBrainState
from .base_agent import BaseAgent
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore",
    )


class RouterAgent(BaseAgent):
    """Agent responsible for routing user queries to the appropriate data source."""

    def __init__(self, llm, websocket=None, stream_output=None, headers=None):
        """Initialize the router agent.
        
        Args:
            llm: The language model to use for routing
            websocket: WebSocket for streaming output
            stream_output: Function to stream output
            headers: Headers for API requests
        """
        super().__init__(websocket, stream_output, headers)
        self.llm = llm
        self.agent_name = "ROUTER"
        self.prompt_template = self._create_prompt_template()
        self.router_chain = self.prompt_template | self.llm
    
    async def run(self, state: CloudBrainState) -> CloudBrainState:
        """
        Main entry point for the router agent.
        
        Args:
            state: The current state of the graph.
            
        Returns:
            The updated state with routing decision.
        """
        # Log the start of routing
        await self._log("Starting query routing", self.agent_name)
        
        try:
            # Extract query from state, handling both dictionary and object access
            if isinstance(state, dict):
                query = state.get("user_query")
            else:
                query = state.user_query
                
            if not query:
                raise ValueError("No user query found in state")
            
            # Log the query being routed
            await self._log(f"Routing query: {query}", self.agent_name)
            
            # Determine the appropriate data source
            route_decision = await self._route_query(query)
            
            # Log the routing decision
            await self._log(f"Routing decision: {route_decision}", self.agent_name)
            
            # Update and return the state
            return self._update_state(state, {
                "route_decision": route_decision
            })
            
        except Exception as e:
            # Log error
            await self._log(f"Error during routing: {e}", self.agent_name)
            
            # Return the state unchanged
            return state
    
    async def _route_query(self, query: str) -> str:
        """Route a query to the appropriate data source.
        
        Args:
            query: The user's query
            
        Returns:
            The routing decision ("vectorstore" or "websearch")
        """
        try:
            # Invoke the router chain
            response = await self.router_chain.ainvoke({
                "question": query
            })
            
            # Extract the routing decision
            route_query = RouteQuery.parse_raw(response.content.strip())
            
            # Map the routing decision to the appropriate agent
            if route_query.datasource == "vectorstore":
                return "RETRIEVE"
            elif route_query.datasource == "websearch":
                return "WEBSEARCH"
            else:
                raise ValueError(f"Invalid datasource: {route_query.datasource}")
                
        except Exception as e:
            # Log error and default to websearch
            await self._log(f"Error parsing routing response: {e}", self.agent_name)
            return "WEBSEARCH"
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for routing.
        
        Returns:
            ChatPromptTemplate for routing
        """
        template = """You are an expert at routing a user question to a vectorstore or web search.
        
        The vectorstore contains documents related to terraform modules. Use the vectorstore for questions on terraform modules. For all else, use web-search.

        User Question: {question}

        Please determine whether to route this question to the vectorstore or web search.
        Respond with a JSON object in this exact format:
        {{"datasource": "vectorstore"}} or {{"datasource": "websearch"}}
        Do not include any other text in your response."""
        
        return ChatPromptTemplate.from_template(template)

# For backward compatibility with AgentRunner
def route(llm, websocket=None, stream_output=None, headers=None):
    """
    Factory function to create a RouterAgent instance.
    
    Args:
        llm: The language model to use for routing
        websocket: WebSocket for streaming output
        stream_output: Function to stream output
        headers: Headers for API requests
        
    Returns:
        A function that takes a state and returns the updated state
    """
    agent = RouterAgent(llm, websocket, stream_output, headers)
    
    async def run(state: CloudBrainState) -> CloudBrainState:
        """
        Run the router agent.
        
        Args:
            state: The current state of the graph.
            
        Returns:
            The updated state with routing decision.
        """
        return await agent.run(state)
    
    return run 