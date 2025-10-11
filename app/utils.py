from pathlib import Path
import yaml


def load_config():
    """
    Load server configs from YAML file.
    """
    # Define path first
    config_path = Path(__file__).parent.parent / "config" / "server_config.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)
