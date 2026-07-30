import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load the project-wide config YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
