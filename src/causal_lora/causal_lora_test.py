import torch
from peft import PeftConfig, PeftModel
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset


def get_inference(batch):
    question = "## Instruction\nPlease answer the brand name of the following website text.\n\n"
    inference = []
    for text, brand in zip(batch["text"], batch["brand"]):
        context = f"## Context\n{text[:1500]}\n\n"  # truncate context 1500 characters
        answer = "## Answer\n"
        prompt = question + context + answer
        inputs = tokenizer(
            prompt, padding=False, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        # Taking logits and converting them to token IDs
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        decoded = tokenizer.decode(predicted_token_ids[0])

        # Extract response content only
        sentinel = "## Answer"
        sentinel_loc = decoded.find(sentinel)
        result = decoded[sentinel_loc + len(sentinel) :]
        inference.append(result)
    return {"inference": inference}


def get_similar_brand(batch):
    # calculate similarity between two brand strings
    query_embeddings = st_model.encode(batch["inference"])

    identified = []
    for query_embedding in query_embeddings:
        brand = brand_list[util.dot_score(query_embedding, passage_embedding).argmax()]
        identified.append(brand)
    return {"identified": identified}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "/content/drive/MyDrive/tuned_models/facebook/opt-350m/checkpoint-5000"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    config = PeftConfig.from_pretrained(model_path)

    model = PeftModel.from_pretrained(model, model_path, config=config).to(device)

    batch_size = 1
    phish_dataset = Dataset.load_from_disk(
        "/content/drive/MyDrive/datasets/phish-text-en"
    )
    dataset = Dataset.from_list(phish_dataset["phish"])
    brand_list = list(set(dataset["brand"]))
    dataset = dataset.remove_columns(["url", "host", "label"])
    test = (
        dataset.shuffle(seed=25)
        .select(range(6000, 6100))
        .map(get_inference, batched=True, batch_size=batch_size)
    )

    st_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    passage_embedding = st_model.encode(brand_list)

    test = test.map(get_similar_brand, batched=True, batch_size=20)

    correct_ans = 0
    for data in test:
        print(
            f"model inference : {data['inference']} / "
            f"identified brand : {data['identified']} / "
            f"correct : {data['brand']}"
        )

        if data["identified"] == data["brand"]:
            correct_ans += 1

    print(f"accuracy : {correct_ans / len(test)}")
