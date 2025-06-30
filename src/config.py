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

    def update(self, key, value):
        """
        Update a value in the configuration by key.

        :param key: The key to update in the configuration.
        :param value: The new value to set for the key.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")

        self.config[key] = value

    def update_from_dict(self, update_dict):
        """
        Update the configuration with key-value pairs from a dictionary.

        :param update_dict: A dictionary containing key-value pairs to update in the configuration.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")

        self.config.update(update_dict)

    def save(self):
        """
        Save the current configuration back to the YAML file.

        :raises yaml.YAMLError: If there is an error writing the YAML file.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")

        with open(self.config_file_path, 'w') as file:
            try:
                yaml.safe_dump(self.config, file, default_flow_style=False)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error writing YAML file: {e}")

    def delete(self, key):
        """
        Delete a key from the configuration.

        :param key: The key to delete from the configuration.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")

        if key in self.config:
            del self.config[key]
