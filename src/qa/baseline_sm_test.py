from argparse import ArgumentParser

from datasets import load_from_disk
from processor import SequenceMatchBrandInferenceProcessor, QADatasetPreprocessor

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="phish-html-en-qa")
    arg_parser.add_argument("--save_mode", type=bool, default=False)
    arg_parser.add_argument(
        "--save_path",
        type=str,
        default="/mnt/d/datasets/phishing_identification/qa_results/baseline/sm_result.csv",
    )

    args = arg_parser.parse_args()
    print("The following arguments are passed:")
    print(args)

    validation_length = 4000
    # load dataset
    base_path = "/mnt/d/datasets/phishing_identification"
    dataset = load_from_disk(f"{base_path}/{args.dataset}").select(
        range(10000, 10000 + validation_length)
    )
    # generate target brand list
    brand_list = list(set(dataset["title"]))

    processor = SequenceMatchBrandInferenceProcessor(brand_list)

    dataset = dataset.map(
        processor.inference_brand_sequence_matcher,
        batched=False, 
        num_proc=20,
    )

    correct_ans = QADatasetPreprocessor.manage_result(
        dataset, save_path=args.save_path, save_mode=args.save_mode
    )

    print(f"the number of Brand List : {len(brand_list)}")
    print(f"accuracy : {correct_ans / validation_length}")
