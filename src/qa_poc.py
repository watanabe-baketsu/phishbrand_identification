from datasets import load_from_disk, Dataset
from transformers import AutoProcessor, MarkupLMForQuestionAnswering, BigBirdTokenizer, BigBirdForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import torch


def inference_brand_markuplm(batch):
    processor = AutoProcessor.from_pretrained("microsoft/markuplm-base-finetuned-websrc")
    model = MarkupLMForQuestionAnswering.from_pretrained("microsoft/markuplm-base-finetuned-websrc").to("cuda")

    question = "What is the brand name of the website?"
    questions = [question] * batch_size

    encoding = processor(batch["html"], questions=questions, return_tensors="pt", padding=True, truncation=True).to("cuda")

    with torch.no_grad():
        outputs = model(**encoding)
    answer_start_index = outputs.start_logits.argmax(dim=1)
    answer_end_index = outputs.end_logits.argmax(dim=1)

    answers = []
    for i in range(batch_size):
        answer_start = answer_start_index[i].item()
        answer_end = answer_end_index[i].item()
        predict_answer_tokens = encoding.input_ids[i, answer_start: answer_end + 1]
        answer = processor.decode(predict_answer_tokens).strip()
        answers.append(answer)
    return {"inference": answers}


# for bigbird
def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]


def inference_brand_bigbird(batch: str):
    model_name = "google/bigbird-base-trivia-itc"
    model = BigBirdForQuestionAnswering.from_pretrained(model_name, attention_type="original_full").to("cuda")
    tokenizer = BigBirdTokenizer.from_pretrained(model_name)

    question = "What is the brand name or title of the website?"
    questions = [question] * batch_size

    encoding = tokenizer(questions, batch["text"], return_tensors='pt', truncation=True, padding=True).to("cuda")
    input_ids = encoding.input_ids.to("cuda")

    with torch.no_grad():
        start_scores, end_scores = model(input_ids=input_ids).to_tuple()

    start_score, end_score = get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100)

    inferences = []
    # Let's convert the input ids back to actual tokens
    for i in range(batch_size):
        all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][i].tolist())
        answer_tokens = all_tokens[start_score: end_score + 1]

        output = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
        print(f"model inference : {output}")
        inferences.append(output)

    return {"inference": inferences}


def get_similar_brand(batch):
    # calculate similarity between two brand strings
    model = SentenceTransformer('all-MiniLM-L6-v2').to("cuda")

    query_embeddings = model.encode(batch["inference"])
    passage_embedding = model.encode(brand_list)

    identified = []
    for query_embedding in query_embeddings:
        brand = brand_list[util.dot_score(query_embedding, passage_embedding).argmax()]
        identified.append(brand)
    # print(f"most similar brand : {brand_list[util.dot_score(query_embedding, passage_embedding).argmax()]} "
    #       f"/ Similarity : {util.dot_score(query_embedding, passage_embedding).max()}")
    return {"identified": identified}


if __name__ == "__main__":
    # load dataset
    dataset = load_from_disk("D:/datasets/phishing_identification/phish-text-en", keep_in_memory=True)
    # generate target brand list
    phish = Dataset.from_list(dataset["phish"])
    brand_list = list(set(phish["brand"]))

    poc_dataset = phish.shuffle(seed=25).select(range(100))
    #poc_dataset = Dataset.from_dict({"phish": poc_dataset})

    batch_size = 2
    poc_dataset = poc_dataset.map(inference_brand_bigbird, batched=True, batch_size=batch_size)
    poc_dataset = poc_dataset.map(get_similar_brand, batched=True, batch_size=batch_size)
    correct_ans = 0
    for data in poc_dataset:
        print(f"model inference : {data['inference']} / "
              f"identified brand : {data['identified']} / "
              f"correct : {data['brand']}")

        if data["identified"] == data["brand"]:
            correct_ans += 1
    # for data in poc_dataset:
    #     html_string = data["html"]
    #     model_inference = inference_brand_markuplm(html_string)
    #
    #     identified_brand = get_similar_brand(model_inference, brand_list)
    #     print(f"model inference : {model_inference} / "
    #           f"identified brand : {identified_brand} / "
    #           f"correct : {data['brand']}")
    #
    #     if identified_brand == data["brand"]:
    #         correct_ans += 1
    #     else:
    #         print(html_string)

    print(f"accuracy : {correct_ans / 100}")
