import os
import torch
from pathlib import Path

CURRENT_DIR = Path(".").resolve()
LOCAL_CACHE_DIR = CURRENT_DIR / "my_local_cache"
HF_HOME = LOCAL_CACHE_DIR / "huggingface"
IRDS_HOME = LOCAL_CACHE_DIR / "ir_datasets"

HF_HOME.mkdir(parents=True, exist_ok=True)
IRDS_HOME.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["IR_DATASETS_HOME"] = str(IRDS_HOME)

print(f"Cache HuggingFace: {HF_HOME}")
print(f"Cache ir_datasets: {IRDS_HOME}")

def get_best_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        print(f"GPU Memory: {total_memory / 1e9:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Found device: {device}")
    return device

device = get_best_device()

OUTPUT_DIR = Path("./outputs/practical-05")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = (OUTPUT_DIR / "index_lotte").absolute()
# Optional extra training pairs (loaded only if the file exists)
SYNTHETIC_DATA_PATH = Path("./outputs/practical-04/training_data_for_splade.json")

# Hugging Face Hub IDs by default; override with environment variables to use local copies
MLM_MODEL_NAME = os.environ.get("MLM_MODEL_NAME", "distilbert-base-uncased")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "HuggingFaceTB/SmolLM2-1.7B-Instruct")

NAVER_MODEL_NAME = os.environ.get("NAVER_MODEL_NAME", "naver/splade-cocondenser-ensembledistil")
NAVER_INDEX_PATH = Path("./index_naver_sota")