import pandas as pd
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util


def inference_brand(batch):
    question = "What is the name of the website's brand?"
    answers = []

    for html in batch["context"]:
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
    similarity = []
    for inference in batch["inference"]:
        query_embedding = st_model.encode(inference)
        sim = util.dot_score(query_embedding, passage_embedding).max()
        similarity.append(sim)
        if sim < 0.5:
            identified_brands.append("other")
        else:
            identified_brands.append(brand_list[util.dot_score(query_embedding, passage_embedding).argmax()])

    return {"identified": identified_brands, "similarity": similarity}


def manage_resulet(targets: Dataset, save_mode=True) -> int:
    correct_ans = 0
    results = []
    if save_mode:
        for data in targets:
            if data["identified"] == data["title"]:
                correct_ans += 1
                is_correct = 1
            else:
                is_correct = 0
            # print(f"answer : {data['title']} / identified : {data['identified']} / similarity : {data['similarity']}")

            # For result analysis
            results.append({
                "inference": data["inference"],
                "identified": data["identified"],
                "similarity": data["similarity"],
                "answer": data["title"],
                "correct": is_correct,
                "html": data["context"]
            })
        result_df = pd.DataFrame(results)
        result_df.to_csv("D:/datasets/phishing_identification/qa_validation_result.csv", index=False)
    else:
        for data in targets:
            if data["identified"] == data["title"]:
                correct_ans += 1

    return correct_ans


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_length = 4000
    # load dataset
    base_path = "D:/datasets/phishing_identification"
    dataset = load_from_disk(f"{base_path}/phish-html-en-qa").select(range(10000,10000+validation_length))
    # generate target brand list
    brand_list = list(set(dataset["title"]))
    model_name = "D:/tuned_models/roberta-base-squad2/checkpoint-5000"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    # calculate similarity between two brand strings
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    passage_embedding = st_model.encode(brand_list)

    dataset = dataset.map(inference_brand, batched=True, batch_size=5)
    dataset = dataset.map(get_similar_brand, batched=True, batch_size=20)

    correct_ans = manage_resulet(dataset, save_mode=False)

    print(f"the number of Brand List : {len(brand_list)}")
    print(f"accuracy : {correct_ans / validation_length}")
