from pathlib import Path


def project_root() -> Path:
    """Project root (parent of backend/)."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    d = project_root()
    return d


def get_artifact_dir() -> Path:
    p = Path(__file__).resolve().parent.parent / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p
