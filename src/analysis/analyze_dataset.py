import pandas as pd
from collections import Counter
from datasets import load_from_disk
from matplotlib import pyplot as plt


class DatasetAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset = load_from_disk(dataset_path, keep_in_memory=True)
        self.labels = list(set(self.dataset["title"]))
        self.df = pd.DataFrame()

    def select_specified_range_samples(self, from_idx: int, to_idx: int):
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

    def get_only_second_label(self, first_from: int, first_to: int, second_from: int, second_to: int) -> list:
        first_dataset = self.dataset.select(range(first_from, first_to))
        second_dataset = self.dataset.select(range(second_from, second_to))

        first_labels = list(set(first_dataset["title"]))
        second_labels = list(set(second_dataset["title"]))

        only_second_labels = []
        for second_label in second_labels:
            if second_label not in first_labels:
                only_second_labels.append(second_label)
        return only_second_labels

    def display_answer_start_mapping(self):
        answer_start_list = []
        for data in self.dataset:
            answer_start_list.append(data["answers"]["answer_start"][0])
        fig, ax = plt.subplots()
        points = [answer_start_list]

        bp = ax.boxplot(points)
        ax.set_xticklabels(['answer start index'])
        plt.grid()
        plt.show()

        # bar plot
        range_size = 100
        counts = Counter((x // range_size) * range_size for x in answer_start_list)

        sorted_counts = sorted(counts.items())
        ranges = [str(x[0]) + "-" + str(x[0] + range_size) for x in sorted_counts]
        counts = [x[1] for x in sorted_counts]

        plt.bar(ranges, counts)
        plt.xlabel('Range')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    analyzer = DatasetAnalyzer("D:/datasets/phishing_identification/phish-html-en-qa")
    only_second_labels = analyzer.get_only_second_label(0, 10000, 10000, 14000)
    print("only second labels : ", only_second_labels)
    analyzer.select_specified_range_samples(10000, 14000)
    print(analyzer.get_num_labels())
    analyzer.display_answer_start_mapping()

    df = analyzer.get_label_percentage()
    # save to csv
    df.to_csv("D:/datasets/phishing_identification/phish-html-en-qa-label-count-training.csv", index=False)
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
