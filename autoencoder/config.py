from pathlib import Path

def force(path: Path) -> Path:
    """
    Force a path to be a directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

ROOT_FOLDER = Path("autoencoder")

MODEL_FOLDER = force(ROOT_FOLDER / "models")
OUTPUT_FOLDER = force(ROOT_FOLDER / "output")

MODEL_PATH = MODEL_FOLDER / "dense_autoencoder_model.keras"
