"""Configuration management for ABC2Vec."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    """
    Manages configuration loading and merging.

    Supports loading from JSON or YAML files with CLI overrides.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize configuration manager.

        Args:
            config_dict: Optional initial configuration dictionary
        """
        self.config = config_dict or {}

    @classmethod
    def from_file(cls, path: str) -> "ConfigManager":
        """
        Load configuration from JSON or YAML file.

        Args:
            path: Path to configuration file

        Returns:
            ConfigManager instance with loaded config
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if path_obj.suffix == ".json":
            with open(path) as f:
                config_dict = json.load(f)
        elif path_obj.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path_obj.suffix}")

        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports nested keys with dots)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary.

        Args:
            updates: Dictionary of updates to apply
        """
        for key, value in updates.items():
            self.set(key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()

    def save(self, path: str) -> None:
        """
        Save configuration to file.

        Args:
            path: Path to save configuration (JSON or YAML)
        """
        path_obj = Path(path)

        if path_obj.suffix == ".json":
            with open(path, "w") as f:
                json.dump(self.config, f, indent=2)
        elif path_obj.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {path_obj.suffix}")
