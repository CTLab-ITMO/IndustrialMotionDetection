import yaml
from pathlib import Path

class YamlConfigReader:
    def __init__(self, config_file_path):
        """
        Initialize the YamlConfigReader with the path to the YAML config file.

        :param config_file_path: Path to the YAML configuration file.
        """
        self.config_file_path = Path(config_file_path)
        self.config = None
        
        self.load()

    def load(self):
        """
        Load and parse the YAML configuration file.

        :raises FileNotFoundError: If the config file does not exist.
        :raises yaml.YAMLError: If there is an error parsing the YAML file.
        """
        if not self.config_file_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file_path}")

        with open(self.config_file_path, 'r') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    def get(self, key, default=None):
        """
        Get a value from the configuration by key.

        :param key: The key to retrieve from the configuration.
        :param default: The default value to return if the key is not found.
        :return: The value associated with the key, or the default value if the key is not found.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")

        return self.config.get(key, default)

    def get_all(self):
        """
        Get the entire configuration as a dictionary.

        :return: The entire configuration dictionary.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")

        return self.config
