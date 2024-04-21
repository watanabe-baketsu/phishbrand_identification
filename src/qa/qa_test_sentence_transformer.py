from argparse import ArgumentParser

import torch
from datasets import load_from_disk

from processor import BrandInferenceProcessor

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--model_name",
        type=str,
        default="/mnt/d/tuned_models/basic/roberta-base-squad2/checkpoint-5000",
    )
    arg_parser.add_argument("--dataset", type=str, default="phish-html-en-qa")
    arg_parser.add_argument("--save_mode", type=bool, default=False)
    arg_parser.add_argument(
        "--save_path",
        type=str,
        default="/mnt/d/datasets/phishing_identification/qa_validation_result.csv",
    )

    args = arg_parser.parse_args()
    print("The following arguments are passed:")
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_length = 4000
    # load dataset
    base_path = "/mnt/d/datasets/phishing_identification"
    dataset = load_from_disk(f"{base_path}/{args.dataset}").select(
        range(10000, 10000 + validation_length)
    )
    # generate target brand list
    brand_list = list(set(dataset["title"]))
    model_name = args.model_name

    processor = BrandInferenceProcessor(model_name, brand_list)

    dataset = dataset.map(processor.inference_brand, batched=True, batch_size=5)
    dataset = dataset.map(processor.get_similar_brand_with_sentence_trandformer, batched=True, batch_size=20)

    correct_ans = processor.manage_result(
        dataset, save_path=args.save_path, save_mode=args.save_mode
    )

    print(f"the number of Brand List : {len(brand_list)}")
    print(f"accuracy : {correct_ans / validation_length}")
