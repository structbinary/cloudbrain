from typing import Union
from typing_extensions import TypedDict


class BaseConfig(TypedDict):
    WEBSEARCH: str
    EMBEDDING: str
    SIMILARITY_THRESHOLD: float
    FAST_LLM: str
    SMART_LLM: str
    STRATEGIC_LLM: str
    FAST_TOKEN_LIMIT: int
    SMART_TOKEN_LIMIT: int
    STRATEGIC_TOKEN_LIMIT: int
    BROWSE_CHUNK_MAX_LENGTH: int
    SUMMARY_TOKEN_LIMIT: int
    TEMPERATURE: float
    USER_AGENT: str
    MAX_SEARCH_RESULTS_PER_QUERY: int
    MEMORY_BACKEND: str
    TOTAL_WORDS: int
    REPORT_FORMAT: str
    CURATE_SOURCES: bool
    MAX_ITERATIONS: int
    LANGUAGE: str
    AGENT_ROLE: Union[str, None]


DEFAULT_CONFIG: BaseConfig = {
    "WEBSEARCH": "tavily",
    "EMBEDDING": "openai:text-embedding-3-small",
    "SIMILARITY_THRESHOLD": 0.42,
    "FAST_LLM": "openai:gpt-4o-mini",
    "SMART_LLM": "openai:gpt-4o-2024-11-20",  
    "STRATEGIC_LLM": "openai:o3-mini", 
    "FAST_TOKEN_LIMIT": 2000,
    "SMART_TOKEN_LIMIT": 4000,
    "STRATEGIC_TOKEN_LIMIT": 4000,
    "BROWSE_CHUNK_MAX_LENGTH": 8192,
    "CURATE_SOURCES": False,
    "SUMMARY_TOKEN_LIMIT": 700,
    "TEMPERATURE": 0.4,
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "MAX_SEARCH_RESULTS_PER_QUERY": 5,
    "MEMORY_BACKEND": "local",
    "TOTAL_WORDS": 1200,
    "REPORT_FORMAT": "APA",
    "MAX_ITERATIONS": 3,
    "AGENT_ROLE": None
}

VALID_RETRIEVERS = [
    "arxiv",
    "bing",
    "custom",
    "duckduckgo",
    "exa",
    "google",
    "searx",
    "semantic_scholar",
    "serpapi",
    "serper",
    "tavily",
    "pubmed_central",
]

# Define supported LLM providers
SUPPORTED_LLM_PROVIDERS = [
    "openai",
    "anthropic",
    "google",
    "azure",
    "ollama",
    "local",
    "custom"
]

# Define supported embedding providers
SUPPORTED_EMBEDDING_PROVIDERS = [
    "openai",
    "cohere",
    "huggingface",
    "sentence-transformers",
    "local",
    "custom"
]