from argparse import ArgumentParser
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk
from processor import QADatasetPreprocessor, BaselineBrandInferenceProcessor


def filter_brands_by_sample_count(dataset, min_sample_count):
    titles = [example["title"] for example in dataset]
    brand_counts = Counter(titles)
    brands_with_enough_samples = [
        brand for brand, count in brand_counts.items() if count >= min_sample_count
    ]
    filtered_dataset = dataset.filter(
        lambda example: example["title"] in brands_with_enough_samples
    )
    return filtered_dataset, brands_with_enough_samples


def evaluate_model(dataset, brands):
    processor = BaselineBrandInferenceProcessor(brands)

    dataset = dataset.map(
        processor.inference_brand_sequence_matcher, batched=True, batch_size=5, num_proc=20
    )

    correct_ans = QADatasetPreprocessor.manage_result(
        dataset, save_path=args.save_path, save_mode=args.save_mode
    )
    accuracy = correct_ans / len(dataset)
    return accuracy


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="phish-html-en-qa")
    arg_parser.add_argument("--save_mode", type=bool, default=False)
    arg_parser.add_argument(
        "--save_path",
        type=str,
        default="/mnt/d/datasets/phishing_identification/qa_results/baseline/only_eval_brands.csv",
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

    only_eval_brands = QADatasetPreprocessor.get_only_eval_brands(
        train_dataset, eval_dataset
    )
    remove_brands = eval_brands - only_eval_brands

    cleaned_eval_dataset = QADatasetPreprocessor.remove_brands_from_dataset(
        eval_dataset, remove_brands
    )
    print(f"評価データセットのみに存在するブランド数: {len(only_eval_brands)}")
    print(f"評価データセットのみに存在するブランド: {only_eval_brands}")

    min_sample_counts = range(1, 16)
    accuracies = []
    filtered_brands_list = []
    for min_sample_count in min_sample_counts:
        filtered_eval_dataset, filtered_brands = filter_brands_by_sample_count(
            cleaned_eval_dataset, min_sample_count
        )
        filtered_brands_list.append(filtered_brands)
        print(
            f"最終的な評価データセットのブランド数 (サンプル数 >= {min_sample_count}): {len(filtered_brands)}"
        )
        print(
            f"最終的な評価データセットのサンプル数 (サンプル数 >= {min_sample_count}): {len(filtered_eval_dataset)}"
        )

        accuracy = evaluate_model(filtered_eval_dataset, filtered_brands)
        accuracies.append(accuracy)
        print(f"サンプル数 >= {min_sample_count} のときの accuracy: {accuracy}")

    # min_sample_countとその時のaccuracy, 全体のデータセットサンプル数とブランド数をcsvに保存
    result_data = {
        "min_sample_count": min_sample_counts,
        "accuracy": accuracies,
        "total_samples": [len(dataset) for dataset in filtered_brands_list],
        "total_brands": [len(brands) for brands in filtered_brands_list],
    }
    result_df = pd.DataFrame(result_data)
    result_df.to_csv("qa_brandsplit_accuracy.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(min_sample_counts, accuracies, marker="o")
    for i, (x, y) in enumerate(zip(min_sample_counts, accuracies)):
        plt.text(x, y + 0.03, f"{y:.2f}", fontsize=10, ha="center", va="bottom")
    plt.xlabel("Minimum Sample Count")
    plt.ylabel("Accuracy")
    x_labels = [
        f"{count}\n(num of brands={len(filtered_brands)})"
        for count, filtered_brands in zip(min_sample_counts, filtered_brands_list)
    ]
    plt.xticks(min_sample_counts, x_labels, rotation=45, ha="right")

    plt.ylim(bottom=0, top=1)  # 縦軸の開始値を0に設定
    plt.grid()
    plt.tight_layout()
    plt.savefig("./baseline_brandsplit_accuracy.pdf", bbox_inches="tight")
