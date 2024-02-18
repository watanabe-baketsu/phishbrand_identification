import torch
from datasets import Dataset, load_from_disk
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer


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

    labels = list(set(testing["title"]))
    labels = labels + ["other"]

    def brand_to_idx(batch):
        return {"brand_idx": [labels.index(title) for title in batch["title"]]}

    training = training.map(brand_to_idx, batched=True)
    testing = testing.map(brand_to_idx, batched=True)

    train_val_dataset = {"train": training, "validation": testing}

    return train_val_dataset


def training_model(st_model: SetFitModel, train_val_dataset: dict) -> SetFitTrainer:
    st_trainer = SetFitTrainer(
        model=st_model,
        train_dataset=Dataset.from_list(train_val_dataset["train"]),
        eval_dataset=Dataset.from_list(train_val_dataset["validation"]),
        loss_class=CosineSimilarityLoss,
        batch_size=16,
        num_iterations=20,
        num_epochs=1,
        metric="accuracy",
        column_mapping={"context": "text", "brand_idx": "label"},
    )
    st_trainer.train()
    metrics = st_trainer.evaluate()

    print(metrics)
    return st_trainer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(
        device
    )
    dataset_path = "D:/datasets/phishing_identification/phish-html-en-qa"
    dataset = load_dataset(dataset_path)

    trainer = training_model(model, dataset)
    trainer.model.save_pretrained(
        save_directory="D:/datasets/phishing_identification/trained_models"
    )
