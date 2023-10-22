from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch


def generate_prompt(batch):
    question = "## Instruction\nPlease answer the brand name of the following website text.\n\n"
    inputs = []
    masks = []
    for text, brand in zip(batch["text"], batch["brand"]):
        context = f"## Context\n{text[:1500]}\n\n"  # truncate context 1500 characters
        answer = f"## Answer\n{brand}\n\n"
        prompt = question + context + answer
        result = tokenizer(prompt, padding=False, truncation=True, max_length=512, return_tensors="pt")
        inputs.append(result["input_ids"].squeeze(0))
        masks.append(result["attention_mask"].squeeze(0))

    return {"input_ids": torch.stack(inputs), "attention_mask": torch.stack(masks)}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    batch_size = 1
    training_args = TrainingArguments(
        output_dir=f"/content/drive/MyDrive/tuned_models/{model_name}",  # output directory
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=30,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        lr_scheduler_type="cosine",
    )

    phish_dataset = Dataset.load_from_disk("/content/drive/MyDrive/datasets/phish-text-en")
    dataset = Dataset.from_list(phish_dataset["phish"])
    dataset = dataset.remove_columns(["url", "host", "label"])
    training = dataset.shuffle(seed=25).select(range(5000)).map(generate_prompt, batched=True, batch_size=batch_size)
    validation = dataset.shuffle(seed=25).select(range(5000, 6000)).map(generate_prompt, batched=True, batch_size=batch_size)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training,
        eval_dataset=validation,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True

    trainer.model.save_pretrained(save_directory="/content/drive/MyDrive/tuned_models")

