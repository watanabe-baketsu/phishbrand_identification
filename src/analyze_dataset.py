import pandas as pd
from datasets import Dataset, load_from_disk
from matplotlib import pyplot as plt


class DatasetAnalyzer:
    def __init__(self, dataset_path: str) -> Dataset:
        self.dataset = load_from_disk(dataset_path, keep_in_memory=True)
        self.labels = list(set(self.dataset["title"]))
        self.df = pd.DataFrame()

    def select_specified_range_samples(self, from_idx: int, to_idx: int) -> Dataset:
        self.dataset = self.dataset.select(range(from_idx, to_idx))
        self.labels = list(set(self.dataset["title"]))

    def get_num_labels(self) -> int:
        return len(self.labels)

    def get_label_percentage(self) -> pd.DataFrame:
        data = []
        for label in self.labels:
            label_count = self.dataset.filter(lambda x: x["title"] == label).num_rows
            label_percentage = label_count / self.dataset.num_rows
            data.append({"label": label, "percentage": label_percentage, "count": label_count})
        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)
        self.df = self.df.sort_values(by="percentage", ascending=False)
        return self.df


if __name__ == "__main__":
    analyzer = DatasetAnalyzer("D:/datasets/phishing_identification/phish-html-en-qa")
    analyzer.select_specified_range_samples(10000, 14000)
    print(analyzer.get_num_labels())
    df = analyzer.get_label_percentage()
    # save to csv
    df.to_csv("D:/datasets/phishing_identification/phish-html-en-qa-label-count.csv", index=False)
    pd.set_option('display.max_rows', 200)
    print(df[df["count"] >= 1])
    fig, ax1 = plt.subplots()

    # 棒グラフ (要素数)
    ax1.bar(df["label"], df["count"], color='b', alpha=0.6, label='Counts')
    ax1.set_ylabel('Counts', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xticks([])

    percentages = df["percentage"].tolist()
    sum_percentages = []
    sum_percentage = 0
    for percentage in percentages:
        sum_percentage += percentage
        sum_percentages.append(sum_percentage)
    # convert list to pandas series
    sum_percentages = pd.Series(sum_percentages)

    ax2 = ax1.twinx()
    ax2.plot(df["label"], sum_percentages, color='r', marker='o', label='Percentage')
    ax2.set_ylabel('Percentage (%)', color='r')
    ax2.tick_params('y', colors='r')

    plt.title("Counts and Percentages of Labels")
    plt.show()
    plt.bar(df["label"], df["percentage"], )
    plt.show()
