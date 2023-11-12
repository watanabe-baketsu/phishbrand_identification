from datasets import load_from_disk

from analysis.analyze_result import ResultAnalyzer


class GPTResultAnalyzer(ResultAnalyzer):
    def __init__(self, result_path: str, mode="csv"):
        super().__init__(result_path, mode)

    def add_correct_column(self):
        # rename column "title" to "answer"
        self.df = self.df.rename(columns={"title": "answer"})
        self.df["correct"] = (self.df["answer"] == self.df["identified"]).astype(int)


def gpt_x_analyze(path: str):
    analyzer = GPTResultAnalyzer(path, "pandas")
    analyzer.add_correct_column()
    print(f"{path.split('/')[-1]} Accuracy : {analyzer.get_accuracy()}")
    df = analyzer.get_metrics_by_brand()
    print(df)
    analyzer.get_summary_plot(df)
    low_metrics_df = analyzer.get_low_metrics_brand(df, 10, 0.8)
    print(f"{path.split('/')[-1]} Low Metrics Brands")
    print(low_metrics_df)


def main():
    gpt_35_result_path = "D:/datasets/phishing_identification/gpt_results/gpt-3.5-turbo-1106-result"
    gpt_4_result_path = "D:/datasets/phishing_identification/gpt_results/gpt-4-1106-preview-result"

    gpt_x_analyze(gpt_35_result_path)
    gpt_x_analyze(gpt_4_result_path)


if __name__ == "__main__":
    main()
