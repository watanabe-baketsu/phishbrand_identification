from collections import Counter
from functools import partial

import pandas as pd
import torch
from datasets import DatasetDict
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class QADatasetPreprocessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_and_align_answers(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    @staticmethod
    def filter_brands(example, brands_to_remove: list) -> bool:
        # どのブランド名も含まない場合にTrueを返す
        return not any(brand in example["title"] for brand in brands_to_remove)

    @staticmethod
    def get_low_sample_brands(
        dataset: DatasetDict, threshold_percentage: float = 10.0
    ) -> list:
        brand_counts = Counter(dataset["title"])
        # サンプル数が少ない順にブランドをソート
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1])

        # 全体のサンプル数
        total_samples = len(dataset)
        # 目標とするサンプル数
        target_samples = total_samples * (threshold_percentage / 100)

        # サンプル数の合計が目標を超えるまでブランドを取り出す
        accumulated_samples = 0
        low_sample_brands = []
        for brand, count in sorted_brands:
            accumulated_samples += count
            low_sample_brands.append(brand)
            if accumulated_samples > target_samples:
                break

        return low_sample_brands

    @staticmethod
    def remove_brands_from_dataset(
        dataset: DatasetDict, brands_to_remove: list
    ) -> DatasetDict:
        filter_function = partial(
            QADatasetPreprocessor.filter_brands, brands_to_remove=brands_to_remove
        )
        filtered_dataset = dataset.filter(filter_function)
        return filtered_dataset


class BrandInferenceProcessor:
    def __init__(self, model, brand_list: list, st_model="all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForQuestionAnswering.from_pretrained(model).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # calculate similarity between two brand strings
        self.st_model = SentenceTransformer(st_model)
        self.brand_list = brand_list
        self.passage_embedding = self.st_model.encode(brand_list)

    def inference_brand(self, batch):
        question = "What is the name of the website's brand?"
        answers = []

        for html in batch["context"]:
            inputs = self.tokenizer(
                question, html, return_tensors="pt", truncation=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs.to(self.device))

            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()

            predict_answer_tokens = inputs.input_ids[
                0, answer_start_index : answer_end_index + 1
            ]
            answer = self.tokenizer.decode(predict_answer_tokens).strip()
            answers.append(answer)
            # print(f"inference : {answer}")

        return {"inference": answers}

    def get_similar_brand_with_sentence_trandformer(self, batch):
        identified_brands = []
        similarity = []
        for inference in batch["inference"]:
            query_embedding = self.st_model.encode(inference)
            sim = util.dot_score(query_embedding, self.passage_embedding).max()
            similarity.append(sim)
            if sim < 0.5:
                identified_brands.append("other")
            else:
                identified_brands.append(
                    self.brand_list[
                        util.dot_score(query_embedding, self.passage_embedding).argmax()
                    ]
                )

        return {"identified": identified_brands, "similarity": similarity}

    def get_similar_brand_with_sequence_matcher(self, batch):
        identified_brands = []
        similarities = []
        for inference in batch["inference"]:
            max_similarity = 0
            most_similar_brand = "other"
            for brand in self.brand_list:
                similarity = SequenceMatcher(None, inference.lower(), brand.lower()).ratio()
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_brand = brand
            similarities.append(max_similarity)
            if max_similarity < 0.5:
                identified_brands.append("other")
            else:
                identified_brands.append(most_similar_brand)

        return {"identified": identified_brands, "similarity": similarities}

    @staticmethod
    def get_only_eval_brands(
        train_dataset: DatasetDict, eval_dataset: DatasetDict
    ) -> set:
        train_bland_list = list(set(train_dataset["title"]))
        remove_brands = QADatasetPreprocessor.get_low_sample_brands(train_dataset, 10)
        train_brands = set(train_bland_list) - set(remove_brands)

        eval_brands = set(eval_dataset["title"])
        only_eval_brands = eval_brands - train_brands
        return only_eval_brands

    @staticmethod
    def manage_result(
        targets: DatasetDict, save_path: str, save_mode: bool = True
    ) -> int:
        correct_ans = 0
        results = []
        if save_mode is True:
            for data in targets:
                if data["identified"] == data["title"]:
                    correct_ans += 1
                    is_correct = 1
                else:
                    is_correct = 0
                # print(f"answer : {data['title']} / identified : {data['identified']} / similarity : {data['similarity']}")

                # For result analysis
                results.append(
                    {
                        "inference": data["inference"],
                        "identified": data["identified"],
                        "similarity": data["similarity"],
                        "answer": data["title"],
                        "correct": is_correct,
                        "html": data["context"],
                    }
                )
            result_df = pd.DataFrame(results)
            print(f"save result to {save_path}")
            result_df.to_csv(save_path, index=False)
        else:
            for data in targets:
                if data["identified"] == data["title"]:
                    correct_ans += 1

        return correct_ans
