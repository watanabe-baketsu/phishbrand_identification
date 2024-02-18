from argparse import ArgumentParser
from collections import Counter

from datasets import load_from_disk

from processor import BrandInferenceProcessor, QADatasetPreprocessor

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--model_name",
        type=str,
        default="/mnt/d/tuned_models/splitbrands/roberta-base-squad2/checkpoint-5000",
    )
    arg_parser.add_argument("--dataset", type=str, default="phish-html-en-qa")
    arg_parser.add_argument("--save_mode", type=bool, default=False)
    arg_parser.add_argument(
        "--save_path",
        type=str,
        default="/mnt/d/datasets/phishing_identification/qa_results/qa_validation_result_brandsplit.csv",
    )

    args = arg_parser.parse_args()
    print("The following arguments are passed:")
    print(args)

    validation_length = 4000
    # load dataset
    base_path = "/mnt/d/datasets/phishing_identification"

    train_dataset = load_from_disk(f"{base_path}/{args.dataset}")
    train_dataset = train_dataset.select(range(10000))

    eval_dataset = load_from_disk(f"{base_path}/{args.dataset}").select(
        range(10000, 10000 + validation_length)
    )

    eval_brands = set(eval_dataset["title"])

    only_eval_brands = BrandInferenceProcessor.get_only_eval_brands(
        train_dataset, eval_dataset
    )
    remove_brands = eval_brands - only_eval_brands

    cleaned_eval_dataset = QADatasetPreprocessor.remove_brands_from_dataset(
        eval_dataset, remove_brands
    )
    print(f"評価データセットのみに存在するブランド数: {len(only_eval_brands)}")
    print(f"評価データセットのみに存在するブランド: {only_eval_brands}")

    # サンプル数が5サンプル以上のブランドに絞り込む
    titles = [example["title"] for example in cleaned_eval_dataset]
    brand_counts = Counter(titles)
    brands_with_enough_samples = [brand for brand, count in brand_counts.items() if count >= 7]
    cleaned_eval_dataset = cleaned_eval_dataset.filter(lambda example: example["title"] in brands_with_enough_samples)
    final_brands = set(cleaned_eval_dataset["title"])
    print(f"最終的な評価データセットのブランド数: {len(final_brands)}")
    print(f"最終的な評価データセットのブランド: {final_brands}")
    print(f"最終的な評価データセットのサンプル数: {len(cleaned_eval_dataset)}")
    only_eval_brands = list(final_brands)

    model_name = args.model_name
    processor = BrandInferenceProcessor(model_name, list(only_eval_brands))

    cleaned_eval_dataset = cleaned_eval_dataset.map(
        processor.inference_brand, batched=True, batch_size=5
    )
    cleaned_eval_dataset = cleaned_eval_dataset.map(
        processor.get_similar_brand, batched=True, batch_size=20
    )

    correct_ans = processor.manage_result(
        cleaned_eval_dataset, save_path=args.save_path, save_mode=args.save_mode
    )

    print(f"the number of Brand List : {len(only_eval_brands)}")
    print(f"accuracy : {correct_ans / len(cleaned_eval_dataset)}")
