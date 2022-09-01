from pathlib import Path

from yaml import load, SafeLoader

with open(Path(Path(__file__).parent, 'config', 'config.yml'), 'r') as config_file:
    config = load(config_file.read(), SafeLoader)
