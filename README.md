# BrandSpotter

Phishing Website Target Brand Identification using Multiple Model Approaches

BrandSpotter is a research framework for identifying target brands in phishing websites using multiple advanced approaches. This repository implements and compares three different methods for brand identification:

1. GPT-based Analysis: Utilizes GPT models to analyze HTML content and extract brand information
2. SetFit Classification: Employs SetFit for efficient few-shot learning of brand identification
3. Causal LoRA: Implements fine-tuned causal language models for brand extraction

## Project Structure

```
src/
├── analysis/         # Analysis tools for model performance evaluation
├── causal_lora/      # Causal LoRA implementation for brand extraction
├── dataset_maker/    # Dataset creation and preprocessing utilities
├── gpt/             # GPT-based brand identification
├── qa/              # Question-Answering components
└── setfit/          # SetFit model for brand classification
```

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
cd src/dataset_maker
python prepare_dataset.py
```

### Model Training and Inference

The project supports three different approaches:

1. GPT-based Analysis:
```bash
cd src/gpt
python gpt_client.py
```

2. SetFit Classification:
```bash
cd src/setfit
python setfit_poc.py
```

3. Causal LoRA:
```bash
cd src/causal_lora
python causal_lora_test.py
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
