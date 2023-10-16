from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import torch


def inference_brand(batch):
    question = "What is the name of the website's brand?"
    answers = []

    for html in batch["html"]:
        inputs = tokenizer(question, html, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model(**inputs.to(device))

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens).strip()
        answers.append(answer)
        # print(f"inference : {answer}")

    return {"inference": answers}


def get_similar_brand(batch):
    identified_brands = []
    for inference in batch["inference"]:
        query_embedding = st_model.encode(inference)
        identified_brands.append(brand_list[util.dot_score(query_embedding, passage_embedding).argmax()])

    return {"identified": identified_brands}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load dataset
    base_path = "D:/datasets/phishing_identification"
    dataset = load_from_disk(f"{base_path}/phish-html-en-qa", keep_in_memory=True).select(range(10000,14000))
    # generate target brand list
    brand_list = list(set(dataset["brand"]))
    model_name = "baketsu/autotrain-xlm-roberta-base-qa-95197146303"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    # calculate similarity between two brand strings
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    passage_embedding = st_model.encode(brand_list)

    poc_dataset = dataset.shuffle(seed=25).select(range(4000))
    poc_dataset = poc_dataset.map(inference_brand, batched=True, batch_size=5)
    poc_dataset = poc_dataset.map(get_similar_brand, batched=True, batch_size=20)

    correct_ans = 0
    for data in poc_dataset:
        if data["identified"] == data["brand"]:
            correct_ans += 1
        # print(f"model inference : {data['inference']} / "
        #       f"identified brand : {data['identified']} / "
        #       f"correct : {data['brand']}")
    print(f"the number of Brand List : {len(brand_list)}")
    print(f"accuracy : {correct_ans / 4000}")
