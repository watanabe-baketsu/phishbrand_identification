import argparse
import os
import torch
from datasets import load_from_disk
from src.qa.processor import QABrandInferenceProcessor, QADatasetPreprocessor
from src.config import MODEL_DIR, PHISH_HTML_EN_QA, QA_RESULT_DIR

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model_name",
        type=str,
        default=os.path.join(MODEL_DIR, "qa", "roberta-base-squad2", "checkpoint-5000"),
    )
    arg_parser.add_argument("--dataset", type=str, default=PHISH_HTML_EN_QA)
    arg_parser.add_argument("--save_mode", type=bool, default=False)
    arg_parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(QA_RESULT_DIR, "st_result", "qa_result.csv"),
    )
    return arg_parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("The following arguments are passed:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # If you don't have MPS (Apple Silicon), you can use CUDA.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_length = 4000
    # load dataset
    dataset = load_from_disk(args.dataset).select(
        range(10000, 10000 + validation_length)
    )
    # generate target brand list
    brand_list = list(set(dataset["title"]))
    model_name = args.model_name

    processor = QABrandInferenceProcessor(model_name, brand_list)

    dataset = dataset.map(
        processor.inference_brand_question_answering, batched=True, batch_size=5
    )
    dataset = dataset.map(
        processor.get_similar_brand_with_sentence_trandformer,
        batched=True,
        batch_size=20,
    )

    correct_ans = QADatasetPreprocessor.manage_result(
        dataset, save_path=args.save_path, save_mode=args.save_mode
    )

    print(f"the number of Brand List : {len(brand_list)}")
    print(f"accuracy : {correct_ans / validation_length}")
