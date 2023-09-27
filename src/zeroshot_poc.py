import torch
from datasets import load_from_disk, Dataset
from transformers import pipeline
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)


if __name__ == "__main__":
    # load dataset
    dataset = load_from_disk("D:/datasets/phishing_identification/phish-text-en", keep_in_memory=True)
    # generate target brand list
    phish = Dataset.from_list(dataset["phish"])
    labels = list(set(phish["brand"]))

    poc_dataset = phish.shuffle(seed=25).select(range(100))

    batch_size = 6
    correct_ans = 0
    for data, out in tqdm(zip(poc_dataset, classifier(poc_dataset["text"], labels, batch_size=batch_size, truncation=True))):

        inference_label = out["labels"][0]
        print(f"model inference : {inference_label} / "
              f"correct : {data['brand']}")
        if inference_label == data["brand"]:
            correct_ans += 1

    print(f"accuracy : {correct_ans / 100}")
