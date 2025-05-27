from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from sentence_transformers.losses import CosineSimilarityLoss

from src.qa.processor import QADatasetPreprocessor
from setfit import SetFitModel, SetFitTrainer
from src.config import MODEL_DIR, PHISH_HTML_EN_QA
import os


def load_dataset(path: str) -> dict:
    qa_dataset = load_from_disk(path)
    training = qa_dataset.select(range(10000))
    testing = qa_dataset.select(range(10000, 14000))
    eval_brands = set(testing["title"])

    low_sample_brands = QADatasetPreprocessor.get_low_sample_brands(training, 10)
    cleaned_training = QADatasetPreprocessor.remove_brands_from_dataset(
        training, low_sample_brands
    )

    only_eval_brands = QADatasetPreprocessor.get_only_eval_brands(training, testing)
    remove_brands = eval_brands - only_eval_brands

    cleaned_testing = QADatasetPreprocessor.remove_brands_from_dataset(
        testing, remove_brands
    )
    target_brands = list(set(cleaned_training["title"] + cleaned_testing["title"]))

    def brand_edit(batch):
        return {
            "title": [
                title if title in target_brands else "other" for title in batch["title"]
            ]
        }

    cleaned_training = cleaned_training.map(brand_edit, batched=True)
    cleaned_testing = cleaned_testing.map(brand_edit, batched=True)

    labels = list(set(cleaned_training["title"] + cleaned_testing["title"]))
    labels = labels + ["other"]

    label_to_brand = {idx: brand for idx, brand in enumerate(labels)}

    def brand_to_idx(batch):
        return {"brand_idx": [labels.index(title) for title in batch["title"]]}

    cleaned_training = cleaned_training.map(brand_to_idx, batched=True)
    cleaned_testing = cleaned_testing.map(brand_to_idx, batched=True)

    train_val_dataset = {"train": cleaned_training, "validation": cleaned_testing}

    return train_val_dataset, label_to_brand


def training_model(st_model: SetFitModel, train_val_dataset: dict) -> SetFitTrainer:
    st_trainer = SetFitTrainer(
        model=st_model,
        train_dataset=Dataset.from_list(train_val_dataset["train"]),
        eval_dataset=Dataset.from_list(train_val_dataset["validation"]),
        loss_class=CosineSimilarityLoss,
        batch_size=16,
        num_iterations=10,
        num_epochs=1,
        metric="accuracy",
        column_mapping={"context": "text", "brand_idx": "label"},
    )
    st_trainer.train()
    metrics = st_trainer.evaluate()

    print(metrics)
    return st_trainer


def evaluate_model(dataset: Dataset) -> float:
    correct_count = 0
    for example in dataset:
        print(f"Title: {example['title']}, Identified: {example['identified']}")
        if example["title"] == example["identified"]:
            correct_count += 1
    return correct_count / len(dataset)


def filter_brands_by_sample_count(dataset, min_sample_count):
    titles = [example["title"] for example in dataset]
    brand_counts = Counter(titles)
    brands_with_enough_samples = [
        brand for brand, count in brand_counts.items() if count >= min_sample_count
    ]
    filtered_dataset = dataset.filter(
        lambda example: example["title"] in brands_with_enough_samples
    )
    return filtered_dataset, brands_with_enough_samples


def evaluate_model_by_sample_count(
    model_path: str, dataset_path: str, label_to_brand: dict
) -> float:
    dataset = load_from_disk(dataset_path)
    validation_length = 4000
    train_dataset = dataset.select(range(10000))

    eval_dataset = dataset.select(range(10000, 10000 + validation_length))

    eval_brands = set(eval_dataset["title"])

    only_eval_brands = QADatasetPreprocessor.get_only_eval_brands(
        train_dataset, eval_dataset
    )
    remove_brands = eval_brands - only_eval_brands

    cleaned_eval_dataset = QADatasetPreprocessor.remove_brands_from_dataset(
        eval_dataset, remove_brands
    )
    print(f"評価データセットのみに存在するブランド数: {len(only_eval_brands)}")
    print(f"評価データセットのみに存在するブランド: {only_eval_brands}")

    model = SetFitModel.from_pretrained(model_path).to("cuda")
    preds_labels = model.predict(cleaned_eval_dataset["context"]).tolist()
    preds = [label_to_brand[label] for label in preds_labels]
    cleaned_eval_dataset = cleaned_eval_dataset.add_column("inference", preds)
    cleaned_eval_dataset = cleaned_eval_dataset.add_column("identified", preds)

    min_sample_counts = range(1, 16)
    accuracies = []
    filtered_brands_list = []
    for min_sample_count in min_sample_counts:
        filtered_eval_dataset, filtered_brands = filter_brands_by_sample_count(
            cleaned_eval_dataset, min_sample_count
        )
        filtered_brands_list.append(filtered_brands)
        print(
            f"最終的な評価データセットのブランド数 (サンプル数 >= {min_sample_count}): {len(filtered_brands)}"
        )
        print(
            f"最終的な評価データセットのサンプル数 (サンプル数 >= {min_sample_count}): {len(filtered_eval_dataset)}"
        )

        accuracy = evaluate_model(filtered_eval_dataset)
        accuracies.append(accuracy)
        print(f"サンプル数 >= {min_sample_count} のときの accuracy: {accuracy}")

    # min_sample_countとその時のaccuracy, 全体のデータセットサンプル数とブランド数をcsvに保存
    result_data = {
        "min_sample_count": min_sample_counts,
        "accuracy": accuracies,
        "total_samples": [len(dataset) for dataset in filtered_brands_list],
        "total_brands": [len(brands) for brands in filtered_brands_list],
    }
    result_df = pd.DataFrame(result_data)
    result_df.to_csv("setfit_brandsplit_accuracy.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(min_sample_counts, accuracies, marker="o")
    for i, (x, y) in enumerate(zip(min_sample_counts, accuracies)):
        plt.text(x, y + 0.03, f"{y:.2f}", fontsize=10, ha="center", va="bottom")
    plt.xlabel("Minimum Sample Count")
    plt.ylabel("Accuracy")
    x_labels = [
        f"{count}\n(num of brands={len(filtered_brands)})"
        for count, filtered_brands in zip(min_sample_counts, filtered_brands_list)
    ]
    plt.xticks(min_sample_counts, x_labels, rotation=45, ha="right")

    plt.ylim(bottom=0, top=1)  # 縦軸の開始値を0に設定
    plt.grid()
    plt.tight_layout()
    plt.savefig("./setfit_brandsplit_accuracy.pdf", bbox_inches="tight")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(
        device
    )
    dataset_path = PHISH_HTML_EN_QA
    dataset, label_to_brand = load_dataset(dataset_path)

    trainer = training_model(model, dataset)

    save_path = os.path.join(MODEL_DIR, "only_eval_brands")
    trainer.model.save_pretrained(save_path)
    evaluate_model_by_sample_count(save_path, dataset_path, label_to_brand)
