import os
import yaml

def get_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Папка config/
    yaml_config_path = os.path.join(base_dir, "config.yaml")
    # yaml_config_path = os.getenv("config.yaml", "config.yaml")
    return yaml.safe_load(open(yaml_config_path))