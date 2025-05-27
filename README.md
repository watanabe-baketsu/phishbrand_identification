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

This project uses Poetry for dependency management. To install the required dependencies:

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

## Usage

### Dataset Preparation

The dataset preparation process includes:
1. HTML content preprocessing
2. Text extraction
3. English content filtering
4. Base64 encoded content removal

```bash
poetry run python -m src.dataset_maker.prepare_dataset
```

### Main QA Model Training and Evaluation

The primary approach uses a QA model for brand identification:

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
    --st_model_name "all-MiniLM-L6-v2" \
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
    --st_model_name "all-MiniLM-L6-v2" \
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
    --st_model_name "all-MiniLM-L6-v2" \
    --save_mode True \
    --save_path "/path/to/save/results_st.csv"
```

### Baseline Methods (For Comparison)

The repository includes several baseline methods for comparison:

1. Sequence Matcher Baseline:
```bash
# Training
poetry run python -m src.qa.baseline_sm_train \
    --dataset "datasets/phish-html-en-qa" \
    --save_dir "/path/to/save/model"

# Testing
poetry run python -m src.qa.baseline_sm_test \
    --dataset "datasets/phish-html-en-qa" \
    --save_mode True \
    --save_path "/path/to/save/results.csv"
```

2. Sentence Transformer Baseline:
```bash
# Training
poetry run python -m src.qa.baseline_st_train \
    --dataset "datasets/phish-html-en-qa" \
    --model_name "all-MiniLM-L6-v2" \
    --save_dir "/path/to/save/model"

# Testing
poetry run python -m src.qa.baseline_st_test \
    --dataset "datasets/phish-html-en-qa" \
    --model_path "/path/to/saved/model" \
    --save_mode True \
    --save_path "/path/to/save/results.csv"
```

3. GPT-based Baseline:
```bash
poetry run python -m src.gpt.gpt_client --dataset_path "datasets/phish-html-en-qa" --output_dir "/path/to/output"
```

Options:
- `--model`: Specify the GPT model to use (default: "gpt-4-1106-preview")
- `--dataset_path`: Path to the dataset (required)
- `--output_dir`: Output directory for results (optional). If not specified, results will be saved to the `gpt_results` folder in the same directory as the dataset.

4. SetFit Baseline:

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

## Contributing

Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.