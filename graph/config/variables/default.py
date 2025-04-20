from typing import Dict, Any
from .base import BaseConfig

class DefaultConfig(BaseConfig):
    """Default configuration for CloudBrain."""
    
    # Override any default values from BaseConfig here
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0
    LLM_MAX_TOKENS = 2000
    LLM_STREAM = True
    
    VECTOR_STORE_PROVIDER = "chroma"
    VECTOR_STORE_COLLECTION = "terraform_docs"
    VECTOR_STORE_EMBEDDING_MODEL = "text-embedding-ada-002"
    VECTOR_STORE_PERSIST_DIR = "./data/chroma"
    EMBEDDING_PROVIDER = "openai"
    
    RETRIEVER = "chroma"
    
    WEBSEARCH_PROVIDER = "tavily"
    WEBSEARCH_API_KEY = None
    WEBSEARCH_MAX_RESULTS = 5
    WEBSEARCH_TIMEOUT = 30
    
    PLANNER_MODEL = "gpt-4"
    PLANNER_TEMPERATURE = 0.7
    RESEARCHER_MODEL = "gpt-4"
    RESEARCHER_TEMPERATURE = 0.7
    WRITER_MODEL = "gpt-4"
    WRITER_TEMPERATURE = 0.7
    REVIEWER_MODEL = "gpt-4"
    REVIEWER_TEMPERATURE = 0.7
    
    GRAPH_MAX_ITERATIONS = 10
    GRAPH_TIMEOUT = 300
    GRAPH_DEBUG = False
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "cloudbrain.log"
    
    TERRAFORM_VERSION = "1.0.0"
    TERRAFORM_PROVIDERS = ["aws", "azurerm", "google"]
    TERRAFORM_MODULE_PATH = "./modules"
    TERRAFORM_STATE_PATH = "./state"
    TERRAFORM_VARIABLES_PATH = "./variables"

# Create a dictionary of default values
DEFAULT_CONFIG: Dict[str, Any] = {
    key: value for key, value in DefaultConfig.__dict__.items()
    if not key.startswith('__') and not callable(value)
}

# Supported providers
SUPPORTED_LLM_PROVIDERS = ["openai", "anthropic"]
SUPPORTED_EMBEDDING_PROVIDERS = ["openai", "chroma"]

# Valid retrievers
VALID_RETRIEVERS = ["chroma", "pinecone", "tavily"] 