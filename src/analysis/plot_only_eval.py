import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # load results
    qa_result_path = "/mnt/d/datasets/phishing_identification/qa_results/st_result/only_eval_result/qa_brandsplit_accuracy.csv"
    baseline_result_path = "/mnt/d/datasets/phishing_identification/qa_results/baseline/only_eval_accuracy.csv"
    qa_results = pd.read_csv(qa_result_path)
    baseline_results = pd.read_csv(baseline_result_path)
    # qa_resultsとbaseline_resultsからaccuracyのみを抽出し、結合する
    qa_accuracies = qa_results[['accuracy']]
    baseline_accuracies = baseline_results[['accuracy']]
    combined_accuracies = pd.concat([qa_accuracies, baseline_accuracies], axis=1)
    combined_accuracies.columns = ['QA_Accuracy', 'Baseline_Accuracy']
    print(combined_accuracies)
    min_sample_counts = qa_results['min_sample_count']
    brand_count = qa_results['total_brands']
    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(min_sample_counts, combined_accuracies['QA_Accuracy'], label='QA Accuracy', marker='o')
    plt.plot(min_sample_counts, combined_accuracies['Baseline_Accuracy'], label='Baseline Accuracy', marker='x')

    # 各点にテキストを追加
    for i, (qa_acc, baseline_acc) in enumerate(zip(combined_accuracies['QA_Accuracy'], combined_accuracies['Baseline_Accuracy'])):
        if i < 10:
            plt.text(i+1, qa_acc + 0.01, f"{qa_acc:.2f}", fontsize=9, ha="center", va="bottom")
        else:
            plt.text(i+1, qa_acc - 0.04, f"{qa_acc:.2f}", fontsize=9, ha="center", va="bottom")
        if i < 10:
            plt.text(i+1, baseline_acc - 0.04, f"{baseline_acc:.2f}", fontsize=9, ha="center", va="bottom")
        else:
            plt.text(i+1, baseline_acc + 0.01, f"{baseline_acc:.2f}", fontsize=9, ha="center", va="bottom")

    plt.xlabel("Sample Index")
    plt.ylabel("Accuracy")
    # x軸のラベル設定
    x_labels = [
        f"{min_count}\n(num of brands={brand_count})"
        for min_count, brand_count in zip(min_sample_counts, brand_count)
    ]
    plt.xticks(min_sample_counts, x_labels, rotation=45, ha="right")
    plt.ylim(0,1.05)  # 縦軸の範囲を0から1に設定
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./comparison_accuracy.pdf", bbox_inches="tight")