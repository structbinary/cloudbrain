import time
import datetime
from langgraph.graph import StateGraph, END
from graph.state import CloudBrainState
from graph.llm.factory import LLMFactory
from graph.config.config import Config
from dotenv import load_dotenv
from graph.tools.factory import SearchFactory
from graph.utils.views import print_agent_output
from graph.agents import GeneratorAgent
from graph.agents import RetrieverAgent
from graph.agents import SearcherAgent
from graph.agents import RouterAgent
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


load_dotenv()

class WorkflowManager:
    """Main class responsible for managing and coordinating the agent workflow"""

    def __init__(self, task: dict, state: CloudBrainState, websocket=None, stream_output=None, tone=None, headers=None):
        self.task = task
        self.state = state
        self.websocket = websocket
        self.stream_output = stream_output
        self.tone = tone
        self.headers = headers
        self.task_id = self._generate_task_id()
        self.config = Config()
        self.llm = LLMFactory.create_llm(**self.config.llm_config)
        self.search = SearchFactory.create_search(**self.config.search_config)
        self.embeddings = LLMFactory.create_embeddings(**self.config.embedding_config)

    def _generate_task_id(self):
        # Currently time based, but can be any unique identifier
        return int(time.time())

    def _initialize_agents(self):
        """Initialize the agents for the graph.
        
        Returns:
            A dictionary of agent instances.
        """
        return {
            "retriever": RetrieverAgent(
                websocket=self.websocket,
                stream_output=self.stream_output,
                headers=self.headers,
                embeddings=self.embeddings,
                config=self.config.embedding_config
            ),
            "generator": GeneratorAgent(
                llm=self.llm,
                websocket=self.websocket,
                stream_output=self.stream_output,
                headers=self.headers
            ),
            "websearch": SearcherAgent(
                search_factory=self.search,
                websocket=self.websocket,
                stream_output=self.stream_output,
                headers=self.headers
            )
        }

    async def _route_question(self, state: CloudBrainState) -> str:
        """
        Helper method to route a question to the appropriate agent.
        
        Args:
            state: The current state of the graph.
            
        Returns:
            The name of the agent to route to.
        """
        # Create a RouterAgent instance
        router_agent = RouterAgent(
            llm=self.llm,
            websocket=self.websocket,
            stream_output=self.stream_output,
            headers=self.headers
        )
        
        # Call the RouterAgent to get the routing decision
        updated_state = await router_agent.run(state)
        
        # Extract the routing decision from the updated state, handling both dictionary and object access
        if isinstance(updated_state, dict):
            route_decision = updated_state.get("route_decision")
        else:
            route_decision = updated_state.route_decision
        
        # Return the appropriate agent name
        if route_decision == "WEBSEARCH":
            return "websearch"
        elif route_decision == "RETRIEVE":
            return "retriever"
        else:
            # Default to websearch if the routing decision is unclear
            print_agent_output("Defaulting to websearch due to unclear routing decision", "ROUTER")
            return "websearch"

    def _create_workflow(self, agents):
        workflow = StateGraph(CloudBrainState)

        # Add nodes for each agent
        workflow.add_node("retriever", agents["retriever"].run)
        workflow.add_node("generator", agents["generator"].run)
        workflow.add_node("websearch", agents["websearch"].run)

        # Add edges
        # Set conditional entry point based on routing decision
        workflow.set_conditional_entry_point(
            self._route_question,
            {
                "websearch": "websearch",
                "retriever": "retriever",
            },
        )
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("websearch", "generator")
        workflow.add_edge("generator", END)

        return workflow

    def init_research_team(self):
        """Initialize and create a workflow for the research team."""
        agents = self._initialize_agents()
        return self._create_workflow(agents)

    async def _log_task_start(self):
        """
        Log the start of a task execution.
        """
        message = f"Starting task execution for query: '{self.task.get('query')}'"
        print_agent_output(message, "MASTER")
        
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "starting_task", message, self.websocket)

    def draw_workflow_graph(self, output_file_path="cloudbrain_graph.png"):
        """
        Draw and save a visualization of the workflow graph.
        
        Args:
            output_file_path: Path where the graph image will be saved.
        """
        try:
            # Initialize the agent workflow
            workflow = self._create_workflow(self._initialize_agents())
            app = workflow.compile()
            
            # Draw the workflow graph using the compiled app
            app.get_graph().draw_mermaid_png(output_file_path=output_file_path)
            print(f"Graph diagram saved to {output_file_path}")
        except Exception as e:
            print(f"Error drawing workflow graph: {e}")
            print("Continuing without graph visualization...")

    async def run_task(self, task_id=None):
        """Run the workflow task with proper async handling and state management.
        
        Args:
            task_id (str, optional): Task ID for tracking. Defaults to None.
            
        Returns:
            dict: Result containing generation and state information
        """
        try:
            # Initialize agents and create workflow
            agents = self._initialize_agents()
            workflow = self._create_workflow(agents)
            compiled_workflow = workflow.compile()
            
            # Use the existing state object instead of creating a new one
            # The state object already contains the user_query from the task
            
            # Log task start
            await self._log_task_start()
            
            # Execute workflow with proper error handling
            try:
                result = await compiled_workflow.ainvoke(self.state)
                
                # Extract generation and state
                generation = result.get("generation", "")
                state = result.get("state", {})
                
                # Convert state to dictionary if it's an object
                if hasattr(state, "__dict__"):
                    state = state.__dict__
                
                return {
                    "generation": generation,
                    "state": state,
                    "task_id": task_id or self.task_id
                }
                
            except Exception as e:
                print(f"Error during workflow execution: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Convert state to dictionary if it's an object
                state_dict = self.state
                if hasattr(state_dict, "__dict__"):
                    state_dict = state_dict.__dict__
                
                return {
                    "generation": f"I encountered an error while processing your request: {str(e)}",
                    "state": state_dict,
                    "task_id": task_id or self.task_id
                }
                
        except Exception as e:
            print(f"Error in run_task: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Convert state to dictionary if it's an object
            state_dict = self.state
            if hasattr(state_dict, "__dict__"):
                state_dict = state_dict.__dict__
            
            return {
                "generation": "I encountered an error while setting up the workflow. Please try again.",
                "state": state_dict,
                "task_id": task_id or self.task_id
            } 