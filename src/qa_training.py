from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer


def preprocess_training_examples(examples):
    questions = ["What is the name of the website's brand?"] * len(examples["html"])
    inputs = tokenizer(
        questions,
        examples["html"],
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["brand_tokens"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_positions.append(examples["start_position"][sample_idx][0])
        end_positions.append(examples["start_position"][sample_idx][0] + len(answer))

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples):
    questions = ["What is the name of the website's brand?"] * len(examples["html"])
    inputs = tokenizer(
        questions,
        examples["html"],
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    for i in range(len(inputs["input_ids"])):
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    return inputs


if __name__ == "__main__":
    model_name = "deepset/roberta-base-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_path = "D:/datasets/phishing_identification"

    # load dataset
    dataset = load_from_disk(f"{base_path}/phish-html-en-qa")
    test_dataset = dataset.select(range(1000))
    dataset = dataset.select(range(1000, len(dataset)-1)).train_test_split(test_size=0.2)
    train_dataset = dataset["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    validation_dataset = dataset["test"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    print(len(dataset["train"]))
    print(len(dataset["test"]))

    batch_size = 10
    args = TrainingArguments(
        output_dir=f"{base_path}/tuned_models/qa_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=20,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics="f1"
    )
    trainer.train()

