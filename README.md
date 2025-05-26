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
poetry run python src/dataset_maker/prepare_dataset.py
```

### Main QA Model Training and Evaluation

The primary approach uses a QA model for brand identification:

1. Train the QA model:
```bash
poetry run python src/qa/qa_training.py \
    --model_name "deepset/roberta-base-squad2" \
    --dataset "phish-html-en-qa" \
    --output_dir "/path/to/output"
```

2. Evaluate the model:

a. Using Sequence Matcher:
```bash
poetry run python src/qa/qa_test_sequence_matcher.py \
    --model_name "/path/to/trained/model" \
    --dataset "phish-html-en-qa" \
    --save_mode True \
    --save_path "/path/to/save/results_sm.csv"
```

b. Using Sentence Transformer:
```bash
poetry run python src/qa/qa_test_sentence_transformer.py \
    --model_name "/path/to/trained/model" \
    --dataset "phish-html-en-qa" \
    --st_model_name "all-MiniLM-L6-v2" \
    --save_mode True \
    --save_path "/path/to/save/results_st.csv"
```

### Baseline Methods (For Comparison)

The repository includes several baseline methods for comparison:

1. Sequence Matcher Baseline:
```bash
# Training
poetry run python src/qa/baseline_sm_train.py \
    --dataset "phish-html-en-qa" \
    --save_dir "/path/to/save/model"

# Testing
poetry run python src/qa/baseline_sm_test.py \
    --dataset "phish-html-en-qa" \
    --save_mode True \
    --save_path "/path/to/save/results.csv"
```

2. Sentence Transformer Baseline:
```bash
# Training
poetry run python src/qa/baseline_st_train.py \
    --dataset "phish-html-en-qa" \
    --model_name "all-MiniLM-L6-v2" \
    --save_dir "/path/to/save/model"

# Testing
poetry run python src/qa/baseline_st_test.py \
    --dataset "phish-html-en-qa" \
    --model_path "/path/to/saved/model" \
    --save_mode True \
    --save_path "/path/to/save/results.csv"
```

3. GPT-based Baseline:
```bash
poetry run python src/gpt/gpt_client.py --dataset_path "/path/to/dataset/phish-html-en-qa" --output_dir "/path/to/output"
```

オプション:
- `--model`: 使用するGPTモデルを指定（デフォルト: "gpt-4-1106-preview"）
- `--dataset_path`: データセットのパス（必須）
- `--output_dir`: 結果の保存先ディレクトリ（オプション）。指定しない場合はデータセットと同じディレクトリの`gpt_results`フォルダに保存されます。

4. SetFit Baseline:
```bash
# Training
poetry run python src/setfit/setfit_train.py \
    --dataset "phish-html-en-qa" \
    --model_name "all-MiniLM-L6-v2" \
    --save_dir "/path/to/save/model"

# Testing
poetry run python src/setfit/setfit_test.py \
    --dataset "phish-html-en-qa" \
    --model_path "/path/to/saved/model" \
    --save_mode True \
    --save_path "/path/to/save/results.csv"
```

5. Causal LoRA Baseline:
```bash
# Training
poetry run python src/causal_lora/causal_lora_train.py \
    --dataset "phish-html-en-qa" \
    --model_name "gpt2" \
    --save_dir "/path/to/save/model"

# Testing
poetry run python src/causal_lora/causal_lora_test.py \
    --dataset "phish-html-en-qa" \
    --model_path "/path/to/saved/model" \
    --save_mode True \
    --save_path "/path/to/save/results.csv"
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