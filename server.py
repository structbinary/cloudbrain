from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import asyncio
import json

from graph.workflow_manager import WorkflowManager
from graph.state import CloudBrainState

app = FastAPI(title="CloudBrain API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    state: Optional[Dict[str, Any]] = None

# Define the async graph function
async def process_request(input_data: Dict[str, Any]):
    print("\n" + "="*50)
    print("REQUEST RECEIVED")
    print(f"Received input data: {input_data}")
    print("="*50 + "\n")
    
    try:
        # Extract messages from input - handle both direct format and CopilotKit GraphQL format
        messages = []
        
        # Check if this is a CopilotKit GraphQL request
        if 'variables' in input_data and 'data' in input_data['variables']:
            copilot_data = input_data['variables']['data']
            if 'messages' in copilot_data:
                # Extract messages from CopilotKit format
                for msg in copilot_data['messages']:
                    if 'textMessage' in msg:
                        text_msg = msg['textMessage']
                        messages.append({
                            "role": text_msg.get('role', 'user'),
                            "content": text_msg.get('content', ''),
                            "name": None
                        })
        
        # If no messages found in CopilotKit format, try direct format
        if not messages and 'messages' in input_data:
            messages = input_data.get("messages", [])
        
        print(f"Processing messages: {messages}")
        
        if not messages:
            print("No messages found in input")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "No message found in the input.",
                    "name": "cloudbrain"
                }],
                "state": {}
            }
        
        # Extract query - get the last user message
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                query = msg.get("content", "")
                break
                
        if not query:
            print("No user query found in messages")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "No user query found in the messages.",
                    "name": "cloudbrain"
                }],
                "state": {}
            }
            
        print(f"Extracted query: {query}")
        
        # Get thread_id from CopilotKit data if available
        thread_id = None
        if 'variables' in input_data and 'data' in input_data['variables']:
            thread_id = input_data['variables']['data'].get('threadId')
        
        task = {
            "query": query,
            "max_iterations": 3,
            "verbose": True,
            "task_id": thread_id
        }
        print(f"Created task: {task}")
        
        # Initialize CloudBrain state and workflow
        print("Initializing WorkflowManager...")
        state = CloudBrainState(
            user_query=query,
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
        workflow_manager = WorkflowManager(task=task, state=state)
        print("WorkflowManager initialized")
        
        # Execute the workflow directly (it's already async)
        print("Executing workflow...")
        result = await workflow_manager.run_task(task_id=task["task_id"])
        print(f"Workflow execution completed: {result}")
        
        # Format the response
        response_content = result.get("generation", "")
        if not response_content:
            print("No generation found in result")
            response_content = "I apologize, but I couldn't generate a response. Please try again."
        
        # Convert the state object to a dictionary if it's not already
        state_dict = result.get("state", {})
        print(f"Original state from result: {state_dict}")
        
        # If state is an object with __dict__ attribute, convert it to a dictionary
        if hasattr(state_dict, "__dict__"):
            state_dict = state_dict.__dict__
            print(f"Converted state to dictionary: {state_dict}")
        
        # If state is still not a dictionary, try to convert it using vars()
        if not isinstance(state_dict, dict):
            try:
                state_dict = vars(state_dict)
                print(f"Converted state using vars(): {state_dict}")
            except:
                print("Failed to convert state using vars()")
                state_dict = {}
        
        # Ensure we have at least some basic state information
        if not state_dict:
            state_dict = {
                "user_query": query,
                "generation": response_content,
                "web_search_needed": False,
                "local_documents": [],
                "web_documents": [],
                "all_documents": [],
                "research_plan": None,
                "research_data": [],
                "human_feedback": None,
                "report_sections": [],
                "title": None,
                "introduction": None,
                "conclusion": None,
                "sources": [],
                "final_report": None,
                "route_decision": None,
                "searches": []
            }
            print(f"Created default state dictionary: {state_dict}")
        
        # Format response to match UI expectations
        response = {
            "messages": [{
                "role": "assistant",
                "content": response_content,
                "name": "cloudbrain"
            }],
            "state": {
                "cloudbrain": state_dict
            },
            "threadId": task.get("task_id", None)
        }
        print(f"Formatted response: {response}")
        
        # Log the response format for debugging
        print(f"Response format: messages={len(response['messages'])}, state={bool(response['state'])}, threadId={response['threadId']}")
        
        return response
        
    except Exception as e:
        print(f"\nERROR in process_request: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n")
        
        return {
            "messages": [{
                "role": "assistant",
                "content": f"I encountered an error while processing your request: {str(e)}",
                "name": "cloudbrain"
            }],
            "state": {
                "cloudbrain": {}
            },
            "threadId": None
        }

@app.post("/api/copilotkit")
async def chat_endpoint(request: Request):
    try:
        # Get raw request body
        body = await request.json()
        print(f"Raw request body: {body}")
        
        # Process the request
        result = await process_request(body)
        
        # Ensure we have a valid response format
        if not result or "messages" not in result:
            result = {
                "messages": [{
                    "role": "assistant",
                    "content": "I apologize, but I couldn't generate a response. Please try again.",
                    "name": "cloudbrain"
                }],
                "state": {
                    "cloudbrain": {}
                },
                "threadId": None
            }
        
        # Log the response for debugging
        print(f"Returning response: {result}")
        
        # Return the response
        return result
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
