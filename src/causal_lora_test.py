import torch
from datasets import Dataset
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util


def generate_prompt(batch):
    question = "## Instruction\nPlease answer the brand name of the following website text.\n\n"
    inputs = []
    masks = []
    for text, brand in zip(batch["text"], batch["brand"]):
        context = f"## Context\n{text[:1500]}\n\n"  # truncate context 1500 characters
        answer = f"## Answer\n"
        prompt = question + context + answer
        result = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt")
        inputs.append(result["input_ids"].squeeze(0))
        masks.append(result["attention_mask"].squeeze(0))

    return {"input_ids": torch.stack(inputs), "attention_mask": torch.stack(masks)}


def get_inference(batch):
    # 推論
    inference = []
    for data in batch["input_ids"]:
        outputs = model.generate(
            input_ids=data,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.75,
            top_k=40,
            no_repeat_ngram_size=2,
        )
        outputs = outputs[0].tolist()
        print(tokenizer.decode(outputs))

        # EOSトークンにヒットしたらデコード完了
        if tokenizer.eos_token_id in outputs:
            eos_index = outputs.index(tokenizer.eos_token_id)
            decoded = tokenizer.decode(outputs[:eos_index])

            # レスポンス内容のみ抽出
            sentinel = "## Answer\n"
            sentinel_loc = decoded.find(sentinel)
            if sentinel_loc >= 0:
                result = decoded[sentinel_loc+len(sentinel):]
            else:
                result = 'Warning: Expected prompt template to be emitted.  Ignoring output.'
        else:
            result = 'Warning: no <eos> detected ignoring output'
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

    model_name = "../tuned_models/google/bigbird-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True)
    config = PeftConfig.from_pretrained(model_name)

    inference_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, is_decoder=True)

    model = PeftModel.from_pretrained(inference_model, model, config=config).to(device)

    batch_size = 2
    phish_dataset = Dataset.load_from_disk("D:/datasets/phishing_identification/phish-text-en")
    dataset = Dataset.from_list(phish_dataset["phish"])
    brand_list = list(set(dataset["brand"]))
    dataset = dataset.remove_columns(["url", "host", "label"])
    test = dataset.shuffle(seed=25).select(range(6000, 7000)).map(generate_prompt, batched=True, batch_size=batch_size)
    test = test.map(get_inference, batched=True, batch_size=batch_size)

    st_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    passage_embedding = st_model.encode(brand_list)

    test = test.map(get_similar_brand, batched=True, batch_size=batch_size)

    correct_ans = 0
    for data in test:
        print(f"model inference : {data['inference']} / "
              f"identified brand : {data['identified']} / "
              f"correct : {data['brand']}")

        if data["identified"] == data["brand"]:
            correct_ans += 1

    print(f"accuracy : {correct_ans / len(test)}")
