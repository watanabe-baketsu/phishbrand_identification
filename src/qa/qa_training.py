import argparse
import os

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from datasets import load_from_disk
from src.config import MODEL_DIR, PHISH_HTML_EN_QA
from src.qa.processor import QADatasetPreprocessor


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model_name", type=str, default="deepset/roberta-base-squad2"
    )
    arg_parser.add_argument("--dataset", type=str, default="phish-html-en-qa")
    arg_parser.add_argument(
        "--output_dir", type=str, default=os.path.join(MODEL_DIR, "qa")
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("The following arguments are passed:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    model_name = args.model_name
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_from_disk(PHISH_HTML_EN_QA)
    dataset = dataset.select(range(10000)).train_test_split(test_size=0.2)

    preprocessor = QADatasetPreprocessor(tokenizer)
    tokenized_dataset = dataset.map(
        preprocessor.tokenize_and_align_answers,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DefaultDataCollator()

    print("Training dataset size:", len(dataset["train"]))
    print("Test dataset size:", len(dataset["test"]))

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
