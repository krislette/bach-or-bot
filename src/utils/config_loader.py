import yaml
from pathlib import Path

# Load config
with open("config/data_config.yml", "r") as f:
    config = yaml.safe_load(f)

BASE_DIR = Path(config["base_dir"]).resolve()

# Resolve paths
DATASET_NPZ = BASE_DIR / config["paths"]["dataset_npz"]
DATASET_CSV = BASE_DIR / config["paths"]["dataset_csv"]
RAW_DIR = BASE_DIR / config["paths"]["raw_dir"]
PROCESSED_DIR = BASE_DIR / config["paths"]["processed_dir"]
PCA_MODEL = BASE_DIR / config["paths"]["pca_path"]