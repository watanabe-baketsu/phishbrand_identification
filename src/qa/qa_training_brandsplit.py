import argparse
import os
from datasets import load_from_disk
from processor import QADatasetPreprocessor
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
from src.config import MODEL_DIR, PHISH_HTML_EN_QA

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name", type=str, default="deepset/roberta-base-squad2")
    arg_parser.add_argument("--dataset", type=str, default=PHISH_HTML_EN_QA)
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(MODEL_DIR, "qa", "splitbrands"),
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

    dataset = load_from_disk(args.dataset).select(range(10000))
    preprocessor = QADatasetPreprocessor(tokenizer)

    # 下から10%のサンプル数の少ないブランドを削除(後のモデルの耐性評価で使用するため)
    remove_brands = preprocessor.get_low_sample_brands(dataset, 10)
    print("削除するブランド:", remove_brands)
    dataset = preprocessor.remove_brands_from_dataset(dataset, remove_brands)
    print("訓練データセットのサイズ:", len(dataset))
    print("削除後のブランド数:", len(set(dataset["title"])))

    dataset = dataset.train_test_split(test_size=0.1)

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
