from pathlib import Path

PROJECT_PATH = Path.cwd()
DATA_PATH = Path(__file__).parents[2] / "data"
MODELS_PATH = Path(__file__).parents[3] / "archive" / "AAE_models"
MODEL_NUMBER_TO_NAME_MAPPING = {"0": "encoder", "1": "decoder", "2": "discriminator"}
