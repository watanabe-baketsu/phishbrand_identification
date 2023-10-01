from datasets import Dataset, load_from_disk
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

from analyze_dataset import DatasetAnalyzer


def load_dataset(dataset_path: str) -> Dataset:
    dataset = load_from_disk(dataset_path)
    dataset = Dataset.from_list(dataset["phish"])

    def brand_edit(batch):
        return {"brand": [brand if brand in target_brands else "other" for brand in batch["brand"]]}

    dataset = dataset.map(brand_edit, batched=True)

    labels = list(set(dataset["brand"]))

    def brand_to_idx(batch):
        return {"brand_idx": [labels.index(brand) for brand in batch["brand"]]}

    dataset = dataset.map(brand_to_idx, batched=True)

    # select sample for training  dataset
    trainings = Dataset.from_list(dataset).shuffle(seed=25).select(range(4000))
    # select sample for validation dataset
    validations = Dataset.from_list(dataset).shuffle(seed=25).select(range(4000, 5000))

    dataset = {"train": trainings, "validation": validations}
    return dataset


def training_model(model: SetFitModel, dataset: Dataset) -> SetFitTrainer:
    trainer = SetFitTrainer(
        model=model,
        train_dataset=Dataset.from_list(dataset["train"]),
        eval_dataset=Dataset.from_list(dataset["validation"]),
        loss_class=CosineSimilarityLoss,
        batch_size=16,
        num_iterations=20,
        num_epochs=1,
        metric="accuracy",
        column_mapping={"text": "text", "brand_idx": "label"},
    )
    trainer.train()
    metrics = trainer.evaluate()

    print(metrics)
    return trainer


if __name__ == "__main__":
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to("cuda")
    dataset_path = "D:/datasets/phishing_identification/phish-text-en"
    analyzer = DatasetAnalyzer(dataset_path)
    analyzer.get_label_percentage()
    target_brands = analyzer.get_upper_count_brands(10)
    dataset = load_dataset(dataset_path)

    trainer = training_model(model, dataset)
    trainer.model.save_pretrained(save_directory="D:/datasets/phishing_identification/trained_models")

