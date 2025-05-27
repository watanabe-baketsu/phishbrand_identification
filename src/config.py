import os
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent

# Base directory for datasets
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")

# Model output directory
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Results output directory
RESULT_DIR = os.path.join(ROOT_DIR, "results")

# Dataset paths
PHISH_HTML_EN_QA = os.path.join(DATASET_DIR, "phish-html-en-qa")
PHISH_HTML_EN = os.path.join(DATASET_DIR, "phish-html-en")
PHISH_HTML_PICKUP_EN = os.path.join(DATASET_DIR, "phish-html-pickup-en")
PHISH_TEXT_EN = os.path.join(DATASET_DIR, "phish-text-en")
PHISH_FULL = os.path.join(DATASET_DIR, "phish-full")
PHISHPEDIA_VANILLA = os.path.join(DATASET_DIR, "phishpedia_vanilla", "phish_sample_30k")

# Result file paths
QA_RESULT_DIR = os.path.join(RESULT_DIR, "qa")
BASELINE_RESULT_DIR = os.path.join(RESULT_DIR, "baseline")
SETFIT_RESULT_DIR = os.path.join(RESULT_DIR, "setfit")
GPT_RESULT_DIR = os.path.join(RESULT_DIR, "gpt")

# Additional dataset paths
BENIGN_SAMPLE_30K = os.path.join(DATASET_DIR, "benign_sample_30k")

# Additional file paths
PHISH_HTML_EN_QA_LONG_JSONL = os.path.join(DATASET_DIR, "phish-html-en-qa-long.jsonl")
TRAINING_JSONL = os.path.join(DATASET_DIR, "training.jsonl")
PHISH_HTML_EN_QA_LABEL_COUNT_CSV = os.path.join(
    DATASET_DIR, "phish-html-en-qa-label-count-training.csv"
)

# GPT result directories
GPT_35_RESULT_DIR = os.path.join(GPT_RESULT_DIR, "gpt-3.5-turbo-1106-result")
GPT_4_RESULT_DIR = os.path.join(GPT_RESULT_DIR, "gpt-4-1106-preview-result")

# Create necessary directories
for dir_path in [
    DATASET_DIR,
    MODEL_DIR,
    RESULT_DIR,
    QA_RESULT_DIR,
    BASELINE_RESULT_DIR,
    SETFIT_RESULT_DIR,
    GPT_RESULT_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)
