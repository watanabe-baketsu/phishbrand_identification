import json

import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util


class BrandIdentifier:
    def __init__(self, brand_list, model) -> None:
        self.brand_list = brand_list
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model).to(self.device)
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.passage_embedding = self.st_model.encode(self.brand_list)

    def inference_brand(self, batch):
        inputs = self.tokenizer(batch["html"], padding=True, truncation=True, return_tensors="pt")

        decoder_input_ids = self.tokenizer(["<pad>"] * len(batch["html"]), return_tensors="pt").input_ids

        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids.to(self.device),
                                 decoder_input_ids=decoder_input_ids.to(self.device))
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        batch["inference"] = self.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

        return {"inference": batch["inference"]}

    def get_similar_brand(self, batch):
        # calculate similarity between two brand strings
        query_embeddings = self.st_model.encode(batch["inference"])

        identified = []
        for query_embedding in query_embeddings:
            brand = self.brand_list[util.dot_score(query_embedding, self.passage_embedding).argmax()]
            identified.append(brand)
        # print(f"most similar brand : {brand_list[util.dot_score(query_embedding, passage_embedding).argmax()]} "
        #       f"/ Similarity : {util.dot_score(query_embedding, passage_embedding).max()}")
        return {"identified": identified}


def generate_training_dataset():
    with open("D:/datasets/phishing_identification/phish-html-en-qa-long.jsonl", "r", encoding="utf-8", errors='ignore') as f:
        contents = f.readlines()
    cnt = 0
    with open("D:/datasets/phishing_identification/phish-html-en-summarization-long.jsonl", "w", encoding="utf-8", errors='ignore') as f:
        for content in contents:
            chunk = json.loads(content)
            data = {
                "text": chunk["context"],
                "target": chunk["answer_text"][0],
            }
            json.dump(data, f)
            f.write("\n")
            cnt += 1
            if cnt == 5000:
                break


if __name__ == "__main__":
    # load dataset
    dataset = load_from_disk("D:/datasets/phishing_identification/phish-html-en-qa", keep_in_memory=True)
    # generate target brand list
    phish = Dataset.from_list(dataset["phish"])
    brand_list = list(set(phish["brand"]))

    poc_dataset = phish.shuffle(seed=25).select(range(100))

    # training_dataset = phish.shuffle(seed=25).select(range(100, 3000))
    # generate_training_dataset(training_dataset)

    model_name = "baketsu/autotrain-phishing_identification-92328144712"
    brand_identifier = BrandIdentifier(brand_list, model_name)

    batch_size = 2
    poc_dataset = poc_dataset.map(brand_identifier.inference_brand, batched=True, batch_size=batch_size)
    poc_dataset = poc_dataset.map(brand_identifier.get_similar_brand, batched=True, batch_size=batch_size)
    correct_ans = 0
    for data in poc_dataset:
        print(f"model inference : {data['inference']} / "
              f"identified brand : {data['identified']} / "
              f"correct : {data['brand']}")

        if data["identified"] == data["brand"]:
            correct_ans += 1

    print(f"accuracy : {correct_ans / 100}")