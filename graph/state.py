from typing import TypedDict, List, Dict, Any, Optional
from langchain.schema import Document
from copilotkit import CopilotKitState


class CloudBrainState(CopilotKitState):
    """
    Represents the state of cloudbrain graph.
    Attributes:
        user_query: The question to search the database with.
        generation: LLM generation
        web_search_needed: Flag indicating if web search is needed (when local vector store doesn't have answers)
        local_documents: list of documents from local vector store (Terraform code)
        web_documents: list of documents from web search
        all_documents: combined list of all documents
        research_plan: The research plan created by the planner
        research_data: Data collected during research
        human_feedback: Feedback from human reviewer
        report_sections: Sections of the final report
        title: Title of the report
        introduction: Introduction section
        conclusion: Conclusion section
        sources: List of sources used
        final_report: The final report
        searches: List of search operations with their status
    """

    user_query: str = ""
    generation: Optional[str] = None
    web_search_needed: bool = False
    local_documents: List[Document] = []
    web_documents: List[Document] = []
    all_documents: List[Document] = []
    research_plan: Optional[str] = None
    research_data: List[Dict[str, Any]] = []
    human_feedback: Optional[str] = None
    report_sections: List[str] = []
    title: Optional[str] = None
    introduction: Optional[str] = None
    conclusion: Optional[str] = None
    sources: List[str] = []
    final_report: Optional[str] = None
    route_decision: Optional[str] = None
    searches: List[Dict[str, Any]] = []
