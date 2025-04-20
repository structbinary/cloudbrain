import json
import os
import warnings
from typing import Dict, Any, List, Union, Type, get_origin, get_args, Optional, Tuple
from .variables.default import DEFAULT_CONFIG, VALID_RETRIEVERS, SUPPORTED_LLM_PROVIDERS, SUPPORTED_EMBEDDING_PROVIDERS
from .variables.base import BaseConfig
import importlib.util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for CloudBrain."""

    CONFIG_DIR = os.path.join(os.path.dirname(__file__), "variables")

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the configuration.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self._config = DEFAULT_CONFIG.copy()
        if config:
            self._config.update(config)
            
        # Set attributes from configuration and environment variables
        self._set_attributes(self._config)
        
        # Handle deprecated attributes
        self._handle_deprecated_attributes()
        
        # Validate the configuration
        self.validate()
            
    # LLM Configuration
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get the LLM configuration."""
        return {
            'provider': self._config.get('LLM_PROVIDER'),
            'model': self._config.get('LLM_MODEL'),
            'temperature': self._config.get('LLM_TEMPERATURE'),
            'max_tokens': self._config.get('LLM_MAX_TOKENS'),
            'stream': self._config.get('LLM_STREAM')
        }
    
    def set_llm_config(self, config: Dict[str, Any]) -> None:
        """Set the LLM configuration.
        
        Args:
            config: LLM configuration dictionary
        """
        for key, value in config.items():
            if key == 'provider':
                self._config['LLM_PROVIDER'] = value
            elif key == 'model':
                self._config['LLM_MODEL'] = value
            elif key == 'temperature':
                self._config['LLM_TEMPERATURE'] = value
            elif key == 'max_tokens':
                self._config['LLM_MAX_TOKENS'] = value
            elif key == 'stream':
                self._config['LLM_STREAM'] = value
    
    def validate_llm_config(self) -> None:
        """Validate the LLM configuration.
        
        Raises:
            ValueError: If the LLM configuration is invalid
        """
        if self.llm_config["provider"] not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {self.llm_config['provider']}")
    
    # Search Configuration
    @property
    def search_config(self) -> Dict[str, Any]:
        """Get the search configuration."""
        config = {
            'provider': self._config.get('WEBSEARCH_PROVIDER'),
            'max_results': self._config.get('WEBSEARCH_MAX_RESULTS'),
            'timeout': self._config.get('WEBSEARCH_TIMEOUT')
        }
        
        # Only add api_key if it exists
        api_key = self._config.get('WEBSEARCH_API_KEY')
        if api_key:
            config['api_key'] = api_key
            
        return config
    
    def set_search_config(self, config: Dict[str, Any]) -> None:
        """Set the search configuration.
        
        Args:
            config: Search configuration dictionary
        """
        for key, value in config.items():
            if key == 'provider':
                self._config['WEBSEARCH_PROVIDER'] = value
            elif key == 'max_results':
                self._config['WEBSEARCH_MAX_RESULTS'] = value
            elif key == 'timeout':
                self._config['WEBSEARCH_TIMEOUT'] = value
            elif key == 'api_key':
                self._config['WEBSEARCH_API_KEY'] = value
    
    # Embedding Configuration
    @property
    def embedding_config(self) -> Dict[str, Any]:
        """Get the embedding configuration."""
        return {
            'provider': self._config.get('EMBEDDING_PROVIDER', 'openai'),
            'model': self._config.get('VECTOR_STORE_EMBEDDING_MODEL'),
            'persist_directory': self._config.get('VECTOR_STORE_PERSIST_DIR', "./vectorstore"),
            'collection_name': self._config.get('VECTOR_STORE_COLLECTION', "terraform_docs")
        }
    
    def set_embedding_config(self, config: Dict[str, Any]) -> None:
        """Set the embedding configuration.
        
        Args:
            config: Embedding configuration dictionary
        """
        for key, value in config.items():
            if key == 'provider':
                self._config['EMBEDDING_PROVIDER'] = value
            elif key == 'model':
                self._config['VECTOR_STORE_EMBEDDING_MODEL'] = value
    
    def validate_embedding_config(self) -> None:
        """Validate the embedding configuration.
        
        Raises:
            ValueError: If the embedding configuration is invalid
        """
        if self.embedding_config["provider"] not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_config['provider']}")
    
    # General Configuration Methods
    def validate(self) -> None:
        """Validate the configuration.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        self.validate_llm_config()
        self.validate_embedding_config()
            
    def update(self, config: Dict[str, Any]) -> None:
        """Update the configuration.
        
        Args:
            config: Configuration dictionary to update with
        """
        self._config.update(config)
        self.validate()

    def _set_attributes(self, config: Dict[str, Any]) -> None:
        """Set attributes from configuration.
        
        Args:
            config: Configuration dictionary
        """
        for key, value in config.items():
            env_value = os.getenv(key)
            if env_value is not None:
                value = self.convert_env_value(key, env_value, BaseConfig.__annotations__[key])
            setattr(self, key.lower(), value)

        # Handle RETRIEVER with default value
        retriever_env = os.environ.get("RETRIEVER", config.get("RETRIEVER", "tavily"))
        try:
            self.retrievers = self.parse_retrievers(retriever_env)
        except ValueError as e:
            print(f"Warning: {str(e)}. Defaulting to 'tavily' retriever.")
            self.retrievers = ["tavily"]

    def _handle_deprecated_attributes(self) -> None:
        """Handle deprecated configuration attributes."""
        if os.getenv("EMBEDDING_PROVIDER") is not None:
            warnings.warn(
                "EMBEDDING_PROVIDER is deprecated. Use VECTOR_STORE_PROVIDER instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.set_embedding_config({"provider": os.environ["EMBEDDING_PROVIDER"]})

        if os.getenv("LLM_PROVIDER") is not None:
            warnings.warn(
                "LLM_PROVIDER is deprecated. Use the new LLM configuration format.",
                FutureWarning,
                stacklevel=2,
            )
            self.set_llm_config({"provider": os.environ["LLM_PROVIDER"]})

        if os.getenv("EMBEDDING_PROVIDER") is not None:
            warnings.warn(
                "EMBEDDING_PROVIDER is deprecated and will be removed soon. Use EMBEDDING instead.",
                FutureWarning,
                stacklevel=2,
            )
            embedding_provider = os.environ["EMBEDDING_PROVIDER"]
            self.set_embedding_config({"provider": embedding_provider})

            match embedding_provider:
                case "ollama":
                    self.set_embedding_config({"model": os.environ["OLLAMA_EMBEDDING_MODEL"]})
                case "custom":
                    self.set_embedding_config({"model": os.getenv("OPENAI_EMBEDDING_MODEL", "custom")})
                case "openai":
                    self.set_embedding_config({"model": "text-embedding-3-large"})
                case "azure_openai":
                    self.set_embedding_config({"model": "text-embedding-3-large"})
                case "huggingface":
                    self.set_embedding_config({"model": "sentence-transformers/all-MiniLM-L6-v2"})
                case "google_genai":
                    self.set_embedding_config({"model": "text-embedding-004"})
                case _:
                    raise Exception("Embedding provider not found.")

        _deprecation_warning = (
            "LLM_PROVIDER, FAST_LLM_MODEL and SMART_LLM_MODEL are deprecated and "
            "will be removed soon. Use FAST_LLM and SMART_LLM instead."
        )
        if os.getenv("FAST_LLM_MODEL") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self._config["FAST_LLM_MODEL"] = os.environ["FAST_LLM_MODEL"]
        if os.getenv("SMART_LLM_MODEL") is not None:
            warnings.warn(_deprecation_warning, FutureWarning, stacklevel=2)
            self._config["SMART_LLM_MODEL"] = os.environ["SMART_LLM_MODEL"]
        
    def _set_doc_path(self, config: Dict[str, Any]) -> None:
        self.doc_path = config['DOC_PATH']
        if self.doc_path:
            try:
                self.validate_doc_path()
            except Exception as e:
                print(f"Warning: Error validating doc_path: {str(e)}. Using default doc_path.")
                self.doc_path = DEFAULT_CONFIG['DOC_PATH']

    @classmethod
    def load_config(cls, config_path: str | None) -> Dict[str, Any]:
        """Load configuration from file or use defaults.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            return DEFAULT_CONFIG

        if not os.path.exists(config_path):
            if config_path and config_path != "default":
                print(f"Warning: Configuration not found at '{config_path}'. Using default configuration.")
            return DEFAULT_CONFIG

        with open(config_path, "r") as f:
            custom_config = json.load(f)

        # Merge with default config
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(custom_config)
        return merged_config

    @classmethod
    def list_available_configs(cls) -> List[str]:
        """List all available configuration names.
        
        Returns:
            List of configuration names
        """
        configs = ["default"]
        for file in os.listdir(cls.CONFIG_DIR):
            if file.endswith(".json"):
                configs.append(file[:-5])
        return configs

    def parse_retrievers(self, retriever_str: str) -> List[str]:
        """Parse the retriever string into a list of retrievers and validate them."""
        retrievers = [retriever.strip()
                      for retriever in retriever_str.split(",")]
        valid_retrievers = self.get_all_retriever_names() or []
        invalid_retrievers = [r for r in retrievers if r not in valid_retrievers]
        if invalid_retrievers:
            raise ValueError(
                f"Invalid retriever(s) found: {', '.join(invalid_retrievers)}. "
                f"Valid options are: {', '.join(valid_retrievers)}."
            )
        return retrievers

    @staticmethod
    def parse_llm(llm_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Parse llm string into (llm_provider, llm_model)."""
        if llm_str is None:
            return None, None
        try:
            provider, model = llm_str.split(":", 1)
            if provider not in SUPPORTED_LLM_PROVIDERS:
                print(f"Warning: Unsupported LLM provider '{provider}'. Supported providers: {', '.join(SUPPORTED_LLM_PROVIDERS)}")
            return provider, model
        except ValueError:
            print(f"Warning: Invalid LLM format '{llm_str}'. Expected format: 'provider:model'")
            return None, None

    @staticmethod
    def parse_embedding(embedding_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Parse embedding string into (embedding_provider, embedding_model)."""
        if embedding_str is None:
            return None, None
        try:
            provider, model = embedding_str.split(":", 1)
            if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
                print(f"Warning: Unsupported embedding provider '{provider}'. Supported providers: {', '.join(SUPPORTED_EMBEDDING_PROVIDERS)}")
            return provider, model
        except ValueError:
            print(f"Warning: Invalid embedding format '{embedding_str}'. Expected format: 'provider:model'")
            return None, None

    def validate_doc_path(self):
        """Ensure that the folder exists at the doc path"""
        os.makedirs(self.doc_path, exist_ok=True)

    @staticmethod
    def convert_env_value(key: str, env_value: str, type_hint: Type) -> Any:
        """Convert environment variable to the appropriate type.
        
        Args:
            key: Configuration key
            env_value: Environment variable value
            type_hint: Type hint for the value
            
        Returns:
            Converted value
        """
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union:
            for arg in args:
                if arg is type(None):
                    if env_value.lower() in ("none", "null", ""):
                        return None
                else:
                    try:
                        return Config.convert_env_value(key, env_value, arg)
                    except ValueError:
                        continue
            raise ValueError(f"Cannot convert {env_value} to any of {args}")

        if type_hint is bool:
            return env_value.lower() in ("true", "1", "yes", "on")
        elif type_hint is int:
            return int(env_value)
        elif type_hint is float:
            return float(env_value)
        elif type_hint in (str, Any):
            return env_value
        elif origin is list or origin is List:
            return json.loads(env_value)
        else:
            raise ValueError(f"Unsupported type {type_hint} for key {key}")

    # Legacy getter methods - these can be deprecated in the future
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration.
        
        Returns:
            LLM configuration dictionary
        """
        return self.llm_config

    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration.
        
        Returns:
            Vector store configuration dictionary
        """
        return self.embedding_config

    def get_web_search_config(self) -> Dict[str, Any]:
        """Get web search configuration.
        
        Returns:
            Web search configuration dictionary
        """
        return self.search_config

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent configuration dictionary
        """
        return {
            "model": getattr(self, f"{agent_name}_model"),
            "temperature": getattr(self, f"{agent_name}_temperature"),
        }

    def get_graph_config(self) -> Dict[str, Any]:
        """Get graph configuration.
        
        Returns:
            Graph configuration dictionary
        """
        return {
            "max_iterations": self.graph_max_iterations,
            "timeout": self.graph_timeout,
            "debug": self.graph_debug,
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.
        
        Returns:
            Logging configuration dictionary
        """
        return {
            "level": self.log_level,
            "format": self.log_format,
            "file": self.log_file,
        }

    def get_terraform_config(self) -> Dict[str, Any]:
        """Get Terraform-specific configuration.
        
        Returns:
            Terraform configuration dictionary
        """
        return {
            "version": self.terraform_version,
            "providers": self.terraform_providers,
            "module_path": self.terraform_module_path,
            "state_path": self.terraform_state_path,
            "variables_path": self.terraform_variables_path,
        }
        
    def check_pkg(self, pkg: str) -> None:
        """Check if a package is installed.
        
        Args:
            pkg: Package name
            
        Raises:
            ImportError: If the package is not installed
        """
        if not importlib.util.find_spec(pkg):
            pkg_kebab = pkg.replace("_", "-")
            raise ImportError(
                f"Unable to import {pkg_kebab}. Please install with "
                f"`pip install -U {pkg_kebab}`"
            )
            
    def get_all_retriever_names(self) -> list:
        """Get a list of all retriever names to be used as validators for supported retrievers.
        
        Returns:
            List of retriever names
        """
        from .variables.default import VALID_RETRIEVERS
        return VALID_RETRIEVERS
