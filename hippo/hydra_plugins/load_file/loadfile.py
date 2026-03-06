from pathlib import Path

from omegaconf import OmegaConf
OmegaConf.register_new_resolver(
    "load_file", lambda filename: Path(filename).read_text().strip()
)
