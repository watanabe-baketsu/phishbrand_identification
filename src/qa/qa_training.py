from argparse import ArgumentParser

from datasets import load_from_disk
from processor import QADatasetPreprocessor
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--model_name", type=str, default="deepset/roberta-base-squad2"
    )
    arg_parser.add_argument("--dataset", type=str, default="phish-html-en-qa")
    arg_parser.add_argument("--output_dir", type=str, default="/mnt/d/tuned_models")
    args = arg_parser.parse_args()

    model_name = args.model_name
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_path = "/mnt/d/datasets/phishing_identification"
    dataset = load_from_disk(f"{base_path}/{args.dataset}")
    dataset = dataset.select(range(10000)).train_test_split(test_size=0.2)

    preprocessor = QADatasetPreprocessor(tokenizer)
    tokenized_dataset = dataset.map(
        preprocessor.tokenize_and_align_answers,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DefaultDataCollator()

    print("訓練データセットのサイズ:", len(dataset["train"]))
    print("テストデータセットのサイズ:", len(dataset["test"]))

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{model_name.split('/')[-1]}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
