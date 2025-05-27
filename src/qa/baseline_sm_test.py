import argparse
import os

from datasets import load_from_disk
from src.config import BASELINE_RESULT_DIR, PHISH_HTML_EN_QA
from src.qa.processor import BaselineBrandInferenceProcessor, QADatasetPreprocessor


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default=PHISH_HTML_EN_QA,
        help="Dataset path",
    )
    arg_parser.add_argument("--save_mode", type=bool, default=False)
    arg_parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(BASELINE_RESULT_DIR, "sm_result.csv"),
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("The following arguments are passed:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    validation_length = 4000
    # load dataset
    dataset = load_from_disk(args.dataset).select(
        range(10000, 10000 + validation_length)
    )
    # generate target brand list
    brand_list = list(set(dataset["title"]))
    print(f"the number of Brand List : {len(brand_list)}")

    processor = BaselineBrandInferenceProcessor(brand_list)

    dataset = dataset.map(
        processor.inference_brand_sequence_matcher,
        batched=True,
        batch_size=200,
        num_proc=20,
    )

    correct_ans = QADatasetPreprocessor.manage_result(
        dataset, save_path=args.save_path, save_mode=args.save_mode
    )

    print(f"the number of Brand List : {len(brand_list)}")
    print(f"accuracy : {correct_ans / validation_length}")
