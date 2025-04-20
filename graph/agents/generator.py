from typing import Dict, Any, List, Optional
from graph.state import CloudBrainState
from .base_agent import BaseAgent
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage


class GeneratorAgent(BaseAgent):
    """Agent responsible for generating Terraform code based on documents and user query."""

    def __init__(self, llm, websocket=None, stream_output=None, headers=None):
        """Initialize the generator agent.
        
        Args:
            llm: The language model to use for generation
            websocket: WebSocket for streaming output
            stream_output: Function to stream output
            headers: Headers for API requests
        """
        super().__init__(websocket, stream_output, headers)
        self.llm = llm
        self.agent_name = "GENERATOR"
        self.prompt_template = self._create_prompt_template()
    
    async def run(self, state: CloudBrainState) -> CloudBrainState:
        """
        Main entry point for the generator agent.
        
        Args:
            state: The current state of the graph.
            
        Returns:
            The updated state with generated code.
        """
        await self._log("Starting code generation", self.agent_name)
        
        try:
            if isinstance(state, dict):
                documents = state.get("all_documents", [])
                query = state.get("user_query")
            else:
                documents = state.all_documents
                query = state.user_query
                
            if not query:
                raise ValueError("No user query found in state")
            await self._log(f"Processing {len(documents)} documents for generation", self.agent_name)
            formatted_docs = self._format_documents(documents)
            generation = await self._generate_terraform_code(query, formatted_docs)
            await self._log("Code generation completed successfully", self.agent_name)
            return self._update_state(state, {
                "generation": generation
            })
            
        except Exception as e:
            await self._log(f"Error during code generation: {e}", self.agent_name)
            return state
    
    async def _generate_terraform_code(self, query: str, documents: str) -> str:
        """Generate Terraform code based on the query and documents.
        
        Args:
            query: The user's query
            documents: Formatted documents to use for generation
            
        Returns:
            Generated Terraform code
        """
        await self._log(f"Generating code for query: {query}", self.agent_name)
        
        prompt = self.prompt_template.format(query=query, documents=documents)
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for use in the prompt.
        
        Args:
            documents: List of documents to format
            
        Returns:
            Formatted document string
        """
        formatted_docs = []
        for i, doc in enumerate(documents):
            formatted_docs.append(f"Document {i+1}:\n{doc.page_content}\n")
        return "\n".join(formatted_docs)
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for code generation.
        
        Returns:
            ChatPromptTemplate for code generation
        """
        template = """You are an expert Terraform developer. Your task is to generate Terraform code based on the user's query and the provided documents.

            User Query: {query}

            Relevant Documents:
            {documents}

        Please generate Terraform code that addresses the user's query. The code should be complete, well-structured, and follow Terraform best practices.

        Terraform Code:
        """
        
        return ChatPromptTemplate.from_template(template)


def generate(llm, websocket=None, stream_output=None, headers=None):
    """Factory function to create a GeneratorAgent instance.
    
    Args:
        llm: The language model to use for generation
        websocket: WebSocket for streaming output
        stream_output: Function to stream output
        headers: Headers for API requests
        
    Returns:
        A function that takes a state and returns an updated state
    """
    agent = GeneratorAgent(llm, websocket, stream_output, headers)
    
    async def run(state: CloudBrainState) -> CloudBrainState:
        """Run the generator agent.
        
        Args:
            state: The current state of the graph.
            
        Returns:
            The updated state with generated code.
        """
        return await agent.run(state)
    
    return run
