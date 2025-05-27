import os

import pandas as pd
import torch
from sentence_transformers.losses import CosineSimilarityLoss

from datasets import Dataset, DatasetDict, load_from_disk
from setfit import SetFitModel, SetFitTrainer
from src.config import MODEL_DIR, PHISH_HTML_EN_QA, SETFIT_RESULT_DIR


def load_dataset(path: str) -> dict:
    qa_dataset = load_from_disk(path)
    training = qa_dataset.select(range(10000))
    testing = qa_dataset.select(range(10000, 14000))
    target_brands = list(set(testing["title"]))

    def brand_edit(batch):
        return {
            "title": [
                title if title in target_brands else "other" for title in batch["title"]
            ]
        }

    training = training.map(brand_edit, batched=True)
    testing = testing.map(brand_edit, batched=True)

    labels = list(set(training["title"] + testing["title"]))
    labels = labels + ["other"]

    label_to_brand = {idx: brand for idx, brand in enumerate(labels)}

    def brand_to_idx(batch):
        return {"brand_idx": [labels.index(title) for title in batch["title"]]}

    training = training.map(brand_to_idx, batched=True)
    testing = testing.map(brand_to_idx, batched=True)

    train_val_dataset = {"train": training, "validation": testing}

    return train_val_dataset, label_to_brand


def training_model(st_model: SetFitModel, train_val_dataset: dict) -> SetFitTrainer:
    st_trainer = SetFitTrainer(
        model=st_model,
        train_dataset=Dataset.from_list(train_val_dataset["train"]),
        eval_dataset=Dataset.from_list(train_val_dataset["validation"]),
        loss_class=CosineSimilarityLoss,
        batch_size=16,
        num_iterations=5,
        num_epochs=1,
        metric="accuracy",
        column_mapping={"context": "text", "brand_idx": "label"},
    )
    st_trainer.train()
    metrics = st_trainer.evaluate()

    print(metrics)
    return st_trainer


def manage_result(targets: DatasetDict, save_path: str, save_mode: bool = True) -> int:
    correct_ans = 0
    results = []
    if save_mode is True:
        for data in targets:
            if data["identified"] == data["title"]:
                correct_ans += 1
                is_correct = 1
            else:
                is_correct = 0
            # print(f"answer : {data['title']} / identified : {data['identified']} / similarity : {data['similarity']}")

            # For result analysis
            results.append(
                {
                    "inference": data["inference"],
                    "identified": data["identified"],
                    "answer": data["title"],
                    "correct": is_correct,
                    "html": data["context"],
                }
            )
        result_df = pd.DataFrame(results)
        print(f"save result to {save_path}")
        result_df.to_csv(save_path, index=False)
    else:
        for data in targets:
            if data["identified"] == data["title"]:
                correct_ans += 1

    return correct_ans


def evaluate_model(model_path: str, dataset_path: str, laebl_to_brand: dict):
    model = SetFitModel.from_pretrained(model_path)
    eval_dataset = load_from_disk(dataset_path).select(range(10000, 14000))

    preds_label = model.predict(eval_dataset["context"]).tolist()
    preds = [label_to_brand[label] for label in preds_label]
    eval_dataset = eval_dataset.add_column("inference", preds)
    eval_dataset = eval_dataset.add_column("identified", preds)

    correct = manage_result(
        eval_dataset,
        os.path.join(SETFIT_RESULT_DIR, "poc_results.csv"),
        save_mode=True,
    )
    print(f"Accuracy: {correct / len(eval_dataset)}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(
        device
    )
    dataset_path = PHISH_HTML_EN_QA
    dataset, label_to_brand = load_dataset(dataset_path)

    trainer = training_model(model, dataset)
    model_path = os.path.join(MODEL_DIR, "vanilla")
    trainer.model.save_pretrained(model_path)

    evaluate_model(model_path, dataset_path, label_to_brand)
