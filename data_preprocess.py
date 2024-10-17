import os
import json

class ConfigLoader:
    """
    Loads configuration data from a JSON file.
    """
    def __init__(self, config_path):
        '''
        Initializes the ConfigLoader object with the specified config file path.

        Parameters:
        ----------
        config_path : str
            Path to the configuration file.

        Returns:
        -------
        No return value.
        '''
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        '''
        Loads the configuration data from the JSON file.

        Returns:
        -------
        dict
            Configuration data loaded from the file.
        '''
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as config_file:
                return json.load(config_file)
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

    def get_config(self):
        '''
        Returns the loaded configuration data.

        Returns:
        -------
        dict
            Loaded configuration data.
        '''
        return self.config