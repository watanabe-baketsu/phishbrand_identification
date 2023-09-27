import torch
from datasets import load_from_disk, Dataset
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=device)


if __name__ == "__main__":
    # load dataset
    dataset = load_from_disk("D:/datasets/phishing_identification/phish-text-en", keep_in_memory=True)
    # generate target brand list
    phish = Dataset.from_list(dataset["phish"])
    labels = list(set(phish["brand"]))

    poc_dataset = phish.shuffle(seed=25).select(range(5000))

    batch_size = 20
    correct_ans = 0
    for data, out in zip(poc_dataset, classifier(poc_dataset["text"], labels, batch_size=batch_size, truncation=True)):

        inference_label = out["labels"][0]
        # print(f"model inference : {inference_label} / "
        #       f"correct : {data['brand']}")
        if inference_label == data["brand"]:
            correct_ans += 1

    print(f"accuracy : {correct_ans / len(poc_dataset)}")
