from typing import Dict, Any, Optional, List
from langchain.schema import Document
from graph.state import CloudBrainState
from graph.utils.views import print_agent_output


class BaseAgent:
    """Base class for all agents in the CloudBrain system."""

    def __init__(self, websocket=None, stream_output=None, headers=None):
        """Initialize the agent.
        
        Args:
            websocket: WebSocket for streaming output
            stream_output: Function to stream output
            headers: Headers for API requests
        """
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
    
    async def _log(self, message: str, agent_name: str = "BASE"):
        """Log a message.
        
        Args:
            message: Message to log
            agent_name: Name of the agent
        """
        if self.websocket and self.stream_output:
            await self.stream_output("logs", agent_name, message, self.websocket)
        else:
            print_agent_output(message, agent_name)
    
    def _update_state(self, state: CloudBrainState, updates: Dict[str, Any]) -> CloudBrainState:
        """Update the state with the given updates.
        
        Args:
            state: Current state
            updates: Updates to apply
            
        Returns:
            Updated state
        """
        # Handle both dictionary and object access
        if isinstance(state, dict):
            # If state is a dictionary, update it directly and return
            return {**state, **updates}
        else:
            # If state is an object, update its attributes
            for key, value in updates.items():
                setattr(state, key, value)
            return state 