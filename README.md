# BrandSpotter

Phishing Website Target Brand Identification using Question-Answering Models

BrandSpotter is a research framework for identifying target brands in phishing websites using advanced question-answering (QA) models. The framework implements a primary QA-based approach along with several baseline methods for comparison.

## Project Structure

```
src/
├── qa/              # Main QA-based brand identification implementation
├── analysis/        # Analysis tools for model performance evaluation
├── dataset_maker/   # Dataset creation and preprocessing utilities
├── gpt/            # GPT-based baseline implementation
├── setfit/         # SetFit baseline implementation
└── causal_lora/    # Causal LoRA baseline implementation

datasets/           # Directory for storing datasets and model outputs
└── phish-html-en-qa/  # Main dataset for phishing brand identification
    ├── *.arrow     # Apache Arrow format cache files for efficient data loading
    ├── data-00000-of-00001.arrow  # Main dataset file (137MB)
    ├── dataset_info.json  # Dataset metadata and configuration
    └── state.json  # Dataset state information

models/             # Pre-trained and fine-tuned models
├── qa/            # Question-Answering models
│   ├── basic/     # Basic QA model (all brands in training)
│   │   └── roberta-base-squad2/
│   │       └── checkpoint-5000/  # RoBERTa-base fine-tuned on SQuAD2 for brand identification
│   └── splitbrand/  # Brand-split QA model (evaluation brands excluded from training)
│       └── roberta-base-squad2/
│           └── checkpoint-5000/  # RoBERTa-base trained with brand split strategy
└── setfit/        # SetFit classification models
    ├── vanilla/   # Standard SetFit model trained on all brands
    └── only_eval_brands/  # SetFit model trained only on evaluation brands
```

## Dataset

### phish-html-en-qa Dataset (`datasets/phish-html-en-qa/`)

The main dataset used for training and evaluating phishing brand identification models.

#### Dataset Structure
- **Format**: Apache Arrow format for efficient data loading and processing
- **Size**: Approximately 1.3GB total (managed via Git LFS)
- **Main Data File**: `data-00000-of-00001.arrow` (137MB)
- **Cache Files**: Multiple `cache-*.arrow` files for optimized data access
- **Metadata**: 
  - `dataset_info.json`: Dataset configuration and schema information
  - `state.json`: Dataset processing state

#### Dataset Features
- **Question**: "What brand is this website imitating?"
- **Context**: Preprocessed HTML content from phishing websites
- **Answer**: Target brand name that the phishing website is imitating
- **Language**: English content only (filtered during preprocessing)

#### Data Preprocessing
The dataset has undergone the following preprocessing steps:
1. HTML content extraction and cleaning
2. English language filtering
3. Base64 encoded content removal
4. Text normalization and tokenization
5. Question-Answer pair generation for QA model training

#### Git LFS Management
This dataset is managed using Git LFS (Large File Storage) for efficient version control:
- All `.arrow` files are tracked by Git LFS
- Enables fast cloning and reduced repository size
- Supports collaborative development with large datasets

#### Usage in Training
The dataset is designed for:
- **Question-Answering Models**: Primary approach for brand identification
- **Classification Models**: Alternative baseline approaches (SetFit, etc.)
- **Generative Models**: GPT-based and Causal LoRA baselines

## Trained Models

### QA Models (`models/qa/`)

#### Basic QA Model (`models/qa/basic/roberta-base-squad2/checkpoint-5000/`)
- **Base Model**: `deepset/roberta-base-squad2`
- **Architecture**: RoBERTaForQuestionAnswering
- **Training**: Trained for 10 epochs, 5000 steps
- **Purpose**: Generate answers to the question "What brand is this website imitating?" from phishing website HTML content
- **Characteristics**: Trained using data from all brands (standard training scenario)

#### Brand-Split QA Model (`models/qa/splitbrand/roberta-base-squad2/checkpoint-5000/`)
- **Base Model**: `deepset/roberta-base-squad2`
- **Architecture**: RoBERTaForQuestionAnswering
- **Training**: Trained on a dataset where evaluation brands are completely excluded from training data
- **Purpose**: Evaluate model performance under challenging conditions where training and evaluation datasets contain completely different brands
- **Characteristics**: Designed to test generalization capability to unseen brands in a realistic zero-shot brand identification scenario

### SetFit Models (`models/setfit/`)

#### Vanilla SetFit Model (`models/setfit/vanilla/`)
- **Base Model**: Sentence Transformer (based on all-MiniLM-L6-v2)
- **Method**: Few-shot learning for text classification
- **Training**: 
  1. Fine-tuning Sentence Transformer with contrastive learning
  2. Training classification head
- **Purpose**: Phishing website brand classification
- **Characteristics**: Standard SetFit approach using data from all brands

#### Evaluation Brands Only SetFit Model (`models/setfit/only_eval_brands/`)
- **Base Model**: Sentence Transformer (based on all-MiniLM-L6-v2)
- **Method**: Few-shot learning for text classification
- **Training**: Trained exclusively on evaluation brands with no overlap between training and evaluation brand sets
- **Purpose**: **Note: This model has limited practical use in the repository**. Users need to manually create their own brand-split datasets and demonstrate model training and evaluation with non-overlapping brand sets between training and evaluation phases
- **Characteristics**: Experimental model for studying brand generalization under strict brand separation conditions

## Installation

This project uses Poetry for dependency management and Git LFS for large file storage.

### Prerequisites

1. **Git LFS**: Required for downloading the dataset files
```bash
# Install Git LFS (if not already installed)
# On macOS with Homebrew:
brew install git-lfs

# On Ubuntu/Debian:
sudo apt install git-lfs

# On Windows: Download from https://git-lfs.github.io/

# Initialize Git LFS
git lfs install
```

2. **Poetry**: For Python dependency management
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -
```

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/BrandSpotter-A-Phishing-Identification-Framework.git
cd BrandSpotter-A-Phishing-Identification-Framework

# Install project dependencies
poetry install

# Verify that the dataset is properly downloaded
ls -la datasets/phish-html-en-qa/
```

**Note**: The first clone may take some time as Git LFS downloads approximately 1.3GB of dataset files.

## Usage

### Dataset Preparation

#### Using the Pre-processed phish-html-en-qa Dataset

The repository includes a pre-processed dataset `datasets/phish-html-en-qa/` that is ready for immediate use:

```bash
# The dataset is automatically available after cloning the repository
# Git LFS will handle the large files automatically

# Verify dataset availability
ls -la datasets/phish-html-en-qa/

# Check dataset info
cat datasets/phish-html-en-qa/dataset_info.json
```

#### Creating New Datasets (Optional)

If you need to create a new dataset from raw HTML files, the dataset preparation process includes:
1. HTML content preprocessing
2. Text extraction
3. English content filtering
4. Base64 encoded content removal
5. Question-Answer pair generation

```bash
poetry run python -m src.dataset_maker.prepare_dataset
```

#### Dataset Loading in Python

Example of loading the phish-html-en-qa dataset in your code:

```python
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("datasets/phish-html-en-qa")

# Access dataset information
print(f"Dataset size: {len(dataset)}")
print(f"Features: {dataset.features}")

# Access a sample
sample = dataset[0]
print(f"Question: {sample['question']}")
print(f"Context: {sample['context'][:200]}...")  # First 200 characters
print(f"Answer: {sample['answers']}")
```

### Main QA Model Training and Evaluation

The primary approach uses a QA model for brand identification. The model is trained to answer the question "What brand is this website imitating?" given HTML content from phishing websites.

#### Quick Start with phish-html-en-qa Dataset

For immediate evaluation using the pre-trained models and phish-html-en-qa dataset:

```bash
# 1. Verify dataset is available
ls -la datasets/phish-html-en-qa/

# 2. Run evaluation with pre-trained basic QA model
poetry run python -m src.qa.qa_test_sequence_matcher \
    --model_name "models/qa/basic/roberta-base-squad2/checkpoint-5000" \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "results/qa_basic_evaluation.csv"

# 3. View results
cat results/qa_basic_evaluation.csv
```

#### Complete Training Pipeline

To train a new QA model from scratch using the phish-html-en-qa dataset:

```bash
# 1. Train a new QA model
poetry run python -m src.qa.qa_training \
    --model_name "deepset/roberta-base-squad2" \
    --dataset "datasets/phish-html-en-qa" \
    --output_dir "models/qa/my_custom_model"

# 2. Evaluate the trained model
poetry run python -m src.qa.qa_test_sequence_matcher \
    --model_name "models/qa/my_custom_model/roberta-base-squad2" \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "results/my_custom_model_results.csv"

# 3. Compare with sentence transformer evaluation
poetry run python -m src.qa.qa_test_sentence_transformer \
    --model_name "models/qa/my_custom_model/roberta-base-squad2" \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "results/my_custom_model_st_results.csv"
```

#### Using Pre-trained Models

To use the pre-trained models:

a. Basic QA Model (trained on all brands):
```bash
# Evaluation using Sequence Matcher
poetry run python -m src.qa.qa_test_sequence_matcher \
    --model_name "models/qa/basic/roberta-base-squad2/checkpoint-5000" \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "results/qa_basic_sm.csv"

# Evaluation using Sentence Transformer
poetry run python -m src.qa.qa_test_sentence_transformer \
    --model_name "models/qa/basic/roberta-base-squad2/checkpoint-5000" \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "results/qa_basic_st.csv"
```

b. Brand-Split QA Model (for challenging cross-brand evaluation):
```bash
# Evaluation using Sequence Matcher
poetry run python -m src.qa.qa_test_sequence_matcher \
    --model_name "models/qa/splitbrand/roberta-base-squad2/checkpoint-5000" \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "results/qa_splitbrand_sm.csv"

# Evaluation using Sentence Transformer
poetry run python -m src.qa.qa_test_sentence_transformer \
    --model_name "models/qa/splitbrand/roberta-base-squad2/checkpoint-5000" \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "results/qa_splitbrand_st.csv"
```

**Note**: The splitbrand model is designed for evaluation under challenging conditions where the brands in training and evaluation datasets do not overlap. This tests the model's ability to generalize to completely unseen brands.

#### Training New Models

To train new models:

1. Train the QA model:
```bash
poetry run python -m src.qa.qa_training \
    --model_name "deepset/roberta-base-squad2" \
    --dataset "datasets/phish-html-en-qa" \
    --output_dir "/path/to/output"
```

2. Evaluate the model:

a. Using Sequence Matcher:
```bash
poetry run python -m src.qa.qa_test_sequence_matcher \
    --model_name "/path/to/trained/model" \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "/path/to/save/results_sm.csv"
```

b. Using Sentence Transformer:
```bash
poetry run python -m src.qa.qa_test_sentence_transformer \
    --model_name "/path/to/trained/model" \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "/path/to/save/results_st.csv"
```

### Baseline Methods (For Comparison)

The repository includes several baseline methods for comparison:

1. Sequence Matcher Baseline:
```bash
# Testing only (no training required for sequence matcher)
poetry run python -m src.qa.baseline_sm_test \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "results/baseline_sm_results.csv"
```

2. GPT-based Baseline:
```bash
poetry run python -m src.gpt.gpt_client --dataset_path "datasets/phish-html-en-qa" --output_dir "/path/to/output"
```

Options:
- `--model`: Specify the GPT model to use (default: "gpt-4-1106-preview")
- `--dataset_path`: Path to the dataset (required)
- `--output_dir`: Output directory for results (optional). If not specified, results will be saved to the `gpt_results` folder in the same directory as the dataset.

3. SetFit Baseline:

#### Using Pre-trained SetFit Models

To use the pre-trained SetFit models:

a. Vanilla SetFit Model (trained on all brands):
```bash
poetry run python -m src.setfit.setfit_test \
    --dataset "datasets/phish-html-en-qa" \
    --model_path "models/setfit/vanilla" \
    --save_mode True \
    --save_path "results/setfit_vanilla.csv"
```

b. Evaluation Brands Only SetFit Model (experimental model with brand separation):
```bash
poetry run python -m src.setfit.setfit_test \
    --dataset "datasets/phish-html-en-qa" \
    --model_path "models/setfit/only_eval_brands" \
    --save_mode True \
    --save_path "results/setfit_eval_brands.csv"
```

**Important Note**: The `only_eval_brands` model has limited practical use in this repository. Users should create their own brand-split datasets and demonstrate model training and evaluation with completely non-overlapping brand sets between training and evaluation phases to properly utilize this approach.

#### Training New SetFit Models

To train new SetFit models:

```bash
# Training
poetry run python -m src.setfit.setfit_train \
    --dataset "datasets/phish-html-en-qa" \
    --model_name "all-MiniLM-L6-v2" \
    --save_dir "/path/to/save/model"

# Testing
poetry run python -m src.setfit.setfit_test \
    --dataset "datasets/phish-html-en-qa" \
    --model_path "/path/to/saved/model" \
    --save_mode True \
    --save_path "/path/to/save/results.csv"
```

5. Causal LoRA Baseline:
```bash
# Training
poetry run python -m src.causal_lora.causal_lora_train \
    --dataset "datasets/phish-html-en-qa" \
    --model_name "gpt2" \
    --save_dir "/path/to/save/model"

# Testing
poetry run python -m src.causal_lora.causal_lora_test \
    --dataset "datasets/phish-html-en-qa" \
    --model_path "/path/to/saved/model" \
    --save_mode True \
    --save_path "/path/to/save/results.csv"
```

### Model Usage Examples

#### QA Model Direct Usage

Example of using the trained QA model directly:

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load model and tokenizer
model_path = "models/qa/basic/roberta-base-squad2/checkpoint-5000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Set question and context
question = "What brand is this website imitating?"
context = "Your HTML content here..."

# Run inference
inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)

# Extract answer
start_scores = outputs.start_logits
end_scores = outputs.end_logits
start_idx = torch.argmax(start_scores)
end_idx = torch.argmax(end_scores)
answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx+1])
```

#### SetFit Model Direct Usage

Example of using the trained SetFit model directly:

```python
from setfit import SetFitModel

# Load model
model = SetFitModel.from_pretrained("models/setfit/vanilla")

# Run inference
texts = ["Your HTML content here..."]
predictions = model(texts)
print(f"Predicted brand: {predictions}")
```

### Analysis

Analysis tools are available in the `src/analysis` directory for evaluating model performance:
- Accuracy, precision, recall, and F1 score calculation
- Brand-specific performance analysis
- Result visualization

## Research Results

The repository includes several PDF files documenting research results:
- `brand_distribution.pdf`: Distribution of target brands in the dataset
- `comparison_accuracy.pdf`: Accuracy comparison between different models
- `language_distribution.pdf`: Distribution of languages in the dataset
- `file_lengths_distribution.pdf`: Distribution of file lengths
- `qa_brandsplit_accuracy.pdf`: QA-based brand split accuracy results
- `baseline_brandsplit_accuracy.pdf`: Baseline model accuracy results
