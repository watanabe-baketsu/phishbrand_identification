from datasets import load_from_disk
from transformers import AutoProcessor, MarkupLMForQuestionAnswering
import torch

processor = AutoProcessor.from_pretrained("microsoft/markuplm-base-finetuned-websrc")
model = MarkupLMForQuestionAnswering.from_pretrained("microsoft/markuplm-base-finetuned-websrc")

dataset = load_from_disk("D:/datasets/phishing_identification/phish")

html_string = dataset["phish"][100]["html"]
question = "What's the brand name of the website?"

encoding = processor(html_string, questions=question, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encoding)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = encoding.input_ids[0, answer_start_index : answer_end_index + 1]
inference_brand = processor.decode(predict_answer_tokens).strip()
print(f"inference : {inference_brand}")
print(f"answer : {dataset['phish'][0]['brand']}")

# calculate similarity between two brand strings
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

query_embedding = model.encode(dataset['phish'][0]['brand'])
passage_embedding = model.encode([inference_brand])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))
