from datasets import load_from_disk, Dataset
from transformers import AutoProcessor, MarkupLMForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import torch


def inference_brand(html_string: str) -> str:
    processor = AutoProcessor.from_pretrained("microsoft/markuplm-base-finetuned-websrc")
    model = MarkupLMForQuestionAnswering.from_pretrained("microsoft/markuplm-base-finetuned-websrc")

    question = "What is the name of the website's brand?"

    encoding = processor(html_string, questions=question, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**encoding)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = encoding.input_ids[0, answer_start_index : answer_end_index + 1]
    answer = processor.decode(predict_answer_tokens).strip()
    # print(f"inference : {answer}")

    return answer


def get_similar_brand(inference_brand: str) -> str:
    # calculate similarity between two brand strings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query_embedding = model.encode(inference_brand)
    passage_embedding = model.encode(brand_list)

    # print(f"most similar brand : {brand_list[util.dot_score(query_embedding, passage_embedding).argmax()]} "
    #       f"/ Similarity : {util.dot_score(query_embedding, passage_embedding).max()}")
    return brand_list[util.dot_score(query_embedding, passage_embedding).argmax()]


if __name__ == "__main__":
    # load dataset
    dataset = load_from_disk("D:/datasets/phishing_identification/phish-html-en-qa", keep_in_memory=True)
    # generate target brand list
    phish = Dataset.from_list(dataset["phish"])
    brand_list = list(set(phish["brand"]))

    poc_dataset = phish.shuffle(seed=25).select(range(100))

    correct_ans = 0
    for data in poc_dataset:
        html_string = data["html"]
        model_inference = inference_brand(html_string)
        identified_brand = get_similar_brand(model_inference, brand_list)
        print(f"model inference : {model_inference} / "
              f"identified brand : {identified_brand} / "
              f"correct : {data['brand']}")

        if identified_brand == data["brand"]:
            correct_ans += 1

    print(f"accuracy : {correct_ans / 100}")