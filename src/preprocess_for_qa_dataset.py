import json

from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer


def tokenize(batch):
    return tokenizer(batch["html"], padding="max_length", truncation=True, return_tensors="pt")


def get_brand_token(batch):
    identified_tokens = []
    start_positions = []
    similarities = []
    for input_ids in batch["input_ids"]:
        passage = []
        for i in range(len(input_ids) - 2):
            decoded_tokens = tokenizer.decode(input_ids[i:i + 3], skip_special_tokens=True)
            passage.append(decoded_tokens)
        passage_embedding = st_model.encode(passage)
        query_embedding = st_model.encode(batch["brand"])
        brand_tokens = passage[util.dot_score(query_embedding, passage_embedding).argmax()]
        identified_tokens.append([brand_tokens])
        start_position = passage.index(brand_tokens)
        start_positions.append([start_position])
        similarity = util.dot_score(query_embedding, passage_embedding).max()
        similarities.append(similarity)
    return {"brand_tokens": identified_tokens, "start_position": start_positions, "similarity": similarities}


def delete_low_similarity_samples(data: Dataset) -> Dataset:
    new_data = []
    for d in data:
        if d["similarity"] > 0.7:
            new_data.append(d)
    return Dataset.from_list(new_data)


def save_sample_dataset_jsonl(data: Dataset):
    cnt = 0
    with open("D:/datasets/phishing_identification/phish-html-en-qa-sample.jsonl", "w", encoding="utf-8", errors='ignore') as f:
        for d in data:
            chunk = {"html": d["html"], "brand_tokens": d["brand_tokens"],
                     "start_position": d["start_position"], "question": "What is the name of the website's brand?"}
            json.dump(chunk, f)
            f.write("\n")
            cnt += 1
            if cnt == 2800:
                break


if __name__ == "__main__":
    # load dataset
    dataset = load_from_disk("D:/datasets/phishing_identification/phish-html-en")
    # generate target brand list
    phish = Dataset.from_list(dataset["phish"]).shuffle()
    brand_list = list(set(phish["brand"]))

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    # load sentence transformer model
    st_model = SentenceTransformer('all-MiniLM-L6-v2')

    # tokenize html
    dataset = phish.map(tokenize, batched=True, batch_size=16)
    # identify brand token
    dataset = dataset.map(get_brand_token, batched=True, batch_size=1)
    dataset = delete_low_similarity_samples(dataset)
    print(f"dataset size : {len(dataset)}")
    print(dataset.column_names)
    dataset.remove_columns(["host", "url", "label"])
    save_sample_dataset_jsonl(dataset)

    dataset.save_to_disk("D:/datasets/phishing_identification/phish-html-en-qa")

    for i in range(10):
        print(f"#### sample{i} : {dataset[i]['brand']}")
        print(dataset[i]["brand_tokens"])
        print(dataset[i]["start_position"])
        print(dataset[i]["similarity"])


