import pandas as pd
from datasets import Dataset, load_from_disk
from matplotlib import pyplot as plt


class DatasetAnalyzer:
    def __init__(self, dataset_path: str) -> Dataset:
        self.dataset = load_from_disk(dataset_path, keep_in_memory=True)
        self.phish = Dataset.from_list(self.dataset["phish"])
        self.labels = list(set(self.phish["brand"]))
        self.df = pd.DataFrame()

    def get_num_labels(self) -> int:
        return len(self.labels)

    def get_label_percentage(self) -> pd.DataFrame:
        data = []
        for label in self.labels:
            label_count = self.phish.filter(lambda x: x["brand"] == label).num_rows
            label_percentage = label_count / self.phish.num_rows
            data.append({"label": label, "percentage": label_percentage, "count": label_count})
        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)
        self.df = self.df.sort_values(by="percentage", ascending=False)
        return self.df

    def get_upper_count_brands(self, cnt: int) -> list:
        target_brands = self.df[self.df["count"] > cnt]["label"].tolist()
        return target_brands


if __name__ == "__main__":
    analyzer = DatasetAnalyzer("D:/datasets/phishing_identification/phish-full")
    print(analyzer.get_num_labels())
    df = analyzer.get_label_percentage()
    print(df[df["count"] > 10])
    print(sum(df[df["count"] > 10]["percentage"]))
    plt.bar(df["label"], df["percentage"])
    plt.show()
