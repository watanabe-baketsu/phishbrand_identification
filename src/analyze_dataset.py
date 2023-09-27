import pandas as pd
from datasets import Dataset, load_from_disk


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


if __name__ == "__main__":
    analyzer = DatasetAnalyzer("D:/datasets/phishing_identification/phish-text-en")
    print(analyzer.get_num_labels())
    df = analyzer.get_label_percentage()
    print(df[df["percentage"] > 0.01])
    print(sum(df[df["percentage"] > 0.01]["percentage"]))
