"""
Configuration Management
========================
Centralized configuration for the OpenNyAI project.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger


class Config:
    """
    Configuration manager for OpenNyAI.
    
    Handles:
    - Environment variables
    - YAML configuration files
    - Project paths
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file.
        """
        # Load environment variables
        load_dotenv()
        
        # Project root
        self.project_root = Path(__file__).parent.parent.parent.parent
        
        # Default paths
        self.data_dir = Path(os.getenv("DATA_DIR", self.project_root / "data"))
        self.raw_data_dir = Path(os.getenv("RAW_DATA_DIR", self.data_dir / "raw"))
        self.processed_data_dir = Path(os.getenv("PROCESSED_DATA_DIR", self.data_dir / "processed"))
        self.model_dir = Path(os.getenv("MODEL_DIR", self.project_root / "models"))
        self.log_dir = Path(os.getenv("LOG_DIR", self.project_root / "logs"))
        
        # Model settings
        self.base_model = os.getenv("BASE_MODEL", "ai4bharat/indic-bert")
        self.max_length = int(os.getenv("MAX_LENGTH", "512"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "16"))
        self.learning_rate = float(os.getenv("LEARNING_RATE", "2e-5"))
        self.num_epochs = int(os.getenv("NUM_EPOCHS", "10"))
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Environment
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "True").lower() == "true"
        
        # Load YAML config if provided
        self.yaml_config = {}
        if config_path:
            self.load_yaml_config(config_path)
        
        logger.info(f"Configuration loaded for {self.environment} environment")
    
    def load_yaml_config(self, config_path: str):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML file.
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        with open(config_path, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
        
        logger.info(f"Loaded YAML config from {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        First checks YAML config, then falls back to environment/defaults.
        
        Args:
            key: Configuration key.
            default: Default value if not found.
            
        Returns:
            Configuration value.
        """
        # Check YAML config first
        if key in self.yaml_config:
            return self.yaml_config[key]
        
        # Check if it's an attribute
        if hasattr(self, key):
            return getattr(self, key)
        
        return default
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-related configuration.
        
        Returns:
            Dictionary with model configuration.
        """
        return {
            "base_model": self.base_model,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
        }
    
    def get_paths(self) -> Dict[str, Path]:
        """
        Get project paths.
        
        Returns:
            Dictionary with project paths.
        """
        return {
            "project_root": self.project_root,
            "data_dir": self.data_dir,
            "raw_data_dir": self.raw_data_dir,
            "processed_data_dir": self.processed_data_dir,
            "model_dir": self.model_dir,
            "log_dir": self.log_dir,
        }
    
    def create_directories(self):
        """Create all necessary project directories."""
        for path in self.get_paths().values():
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created project directories")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "model_config": self.get_model_config(),
            "paths": {k: str(v) for k, v in self.get_paths().items()},
        }


# Global config instance
config = Config()


if __name__ == "__main__":
    # Example usage
    cfg = Config()
    print(f"Project root: {cfg.project_root}")
    print(f"Environment: {cfg.environment}")
    print(f"Model config: {cfg.get_model_config()}")
