from typing import Dict, Any, Optional
from graph.workflow_manager import WorkflowManager
from graph.state import CloudBrainState

class ChiefEditorAgent:
    """
    Main agent class that acts as a chief editor for the CloudBrain system.
    This class provides a simple interface for initializing, compiling, and running agents.
    """

    def __init__(
        self,
        task: Dict[str, Any],
        websocket=None,
        stream_output=None,
        tone=None,
        headers=None
    ):
        """
        Initialize the ChiefEditorAgent.
        
        Args:
            task: A dictionary containing the task details.
            websocket: WebSocket for streaming output.
            stream_output: Function to stream output.
            tone: Tone to use for generation.
            headers: Headers for API requests.
        """
        self.task = task
        self.websocket = websocket
        self.stream_output = stream_output
        self.tone = tone
        self.headers = headers
        
        # Initialize the state
        self.state = CloudBrainState(
            user_query=task.get("query", ""),
            generation=None,
            web_search_needed=False,
            local_documents=[],
            web_documents=[],
            all_documents=[],
            research_plan=None,
            research_data=[],
            human_feedback=None,
            report_sections=[],
            title=None,
            introduction=None,
            conclusion=None,
            sources=[],
            final_report=None,
            route_decision=None,
            searches=[]
        )
        
        # Initialize the workflow manager
        self.workflow_manager = WorkflowManager(
            task=task,
            state=self.state,
            websocket=websocket,
            stream_output=stream_output,
            tone=tone,
            headers=headers
        )
    
    def init_workflow(self):
        """
        Initialize the agent workflow.
        
        Returns:
            The compiled workflow.
        """
        return self.workflow_manager.init_research_team()
    
    def draw_workflow_graph(self, output_file_path="cloudbrain_graph.png"):
        """
        Draw and save a visualization of the workflow graph.
        
        Args:
            output_file_path: Path where the graph image will be saved.
        """
        print(f"Attempting to draw workflow graph to {output_file_path}...")
        try:
            self.workflow_manager.draw_workflow_graph(output_file_path)
            print(f"Workflow graph drawing completed.")
        except Exception as e:
            print(f"Error in ChiefEditorAgent.draw_workflow_graph: {e}")
            print("Continuing without graph visualization...")
    
    async def run(self, task_id=None):
        """
        Run the agent workflow.
        
        Args:
            task_id (optional): The ID of the task to run.
            
        Returns:
            The result of the task execution.
        """
        return await self.workflow_manager.run_task(task_id)
    


# Define default task for the workflow
default_task = {
    "query": "can you write a terraform module for aws s3",
    "max_iterations": 3,
    "verbose": True
}

# Create the agent with default task
agent = ChiefEditorAgent(default_task)

# Initialize state with default query
initial_state = CloudBrainState(
    user_query=default_task["query"],
    generation=None,
    web_search_needed=False,
    local_documents=[],
    web_documents=[],
    all_documents=[],
    research_plan=None,
    research_data=[],
    human_feedback=None,
    report_sections=[],
    title=None,
    introduction=None,
    conclusion=None,
    sources=[],
    final_report=None,
    route_decision=None,
    searches=[]
)

# Initialize and compile the workflow with initial state
workflow = agent.init_workflow()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Define the task
    task = {
        "query": "who is sachin tendulkar",
        "max_iterations": 3,
        "verbose": True
    }
    
    # Create the agent
    agent = ChiefEditorAgent(task)
    
    # Run the agent
    async def main():
        # First run the agent
        result = await agent.run()
        print("Task completed:", result)
        
        # Then draw the workflow graph after the task is completed
        print("Drawing workflow graph...")
        agent.draw_workflow_graph("cloudbrain_graph.png")
        print("Workflow graph drawing completed.")
    
    asyncio.run(main()) 