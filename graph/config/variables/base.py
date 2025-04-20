from typing import Dict, Any, List, Union, Optional

class BaseConfig:
    """Base configuration class with type hints."""
    
    # LLM Configuration
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2000
    LLM_STREAM: bool = True
    
    # Vector Store Configuration
    VECTOR_STORE_PROVIDER: str = "chroma"
    VECTOR_STORE_COLLECTION: str = "terraform_docs"
    VECTOR_STORE_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    VECTOR_STORE_PERSIST_DIR: str = "./data/chroma"
    
    # Web Search Configuration
    WEB_SEARCH_PROVIDER: str = "serpapi"
    WEB_SEARCH_RESULTS: int = 5
    WEB_SEARCH_TIMEOUT: int = 30
    
    # Agent Configuration
    PLANNER_MODEL: str = "gpt-4"
    PLANNER_TEMPERATURE: float = 0.7
    RESEARCHER_MODEL: str = "gpt-4"
    RESEARCHER_TEMPERATURE: float = 0.7
    WRITER_MODEL: str = "gpt-4"
    WRITER_TEMPERATURE: float = 0.7
    REVIEWER_MODEL: str = "gpt-4"
    REVIEWER_TEMPERATURE: float = 0.7
    
    # Graph Configuration
    GRAPH_MAX_ITERATIONS: int = 10
    GRAPH_TIMEOUT: int = 300
    GRAPH_DEBUG: bool = False
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "cloudbrain.log"
    
    # Terraform Specific Configuration
    TERRAFORM_VERSION: str = "1.0.0"
    TERRAFORM_PROVIDERS: List[str] = ["aws", "azurerm", "google"]
    TERRAFORM_MODULE_PATH: str = "./modules"
    TERRAFORM_STATE_PATH: str = "./state"
    TERRAFORM_VARIABLES_PATH: str = "./variables"
    
    # API Keys and Credentials (to be set via environment variables)
    OPENAI_API_KEY: Optional[str] = None
    SERPAPI_API_KEY: Optional[str] = None
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None 