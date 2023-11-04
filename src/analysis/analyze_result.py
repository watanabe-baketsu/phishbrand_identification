import pandas as pd
import matplotlib.pyplot as plt


class ResultAnalyzer:
    def __init__(self, result_path: str):
        self.df = pd.read_csv(result_path)

    def get_accuracy(self) -> float:
        return self.df["correct"].sum() / self.df.shape[0]

    def get_metrics_by_brand(self) -> pd.DataFrame:
        # calculate recall by brand
        df_recall = self.df.groupby("answer").agg({"correct": ["sum", "count"]})
        df_recall.columns = ["correct_sum", "count"]
        df_recall["recall"] = df_recall["correct_sum"] / df_recall["count"]
        df_recall = df_recall.drop(["correct_sum"], axis=1)
        # calculate precision by brand
        df_precision = self.df.groupby("identified").agg({"correct": "sum", "identified": "count"})
        df_precision.columns = ["correct_sum", "identified_count"]
        df_precision["precision"] = df_precision["correct_sum"] / df_precision["identified_count"]
        df_precision = df_precision.drop(["correct_sum", "identified_count"], axis=1)

        df = pd.merge(
            df_recall[["recall", "count"]], df_precision[["precision"]],
            left_index=True,
            right_index=True,
            how="outer"
        )
        df = df.fillna(0)
        df["f1"] = 2 * (df["precision"] * df["recall"]) / (df["precision"] + df["recall"])
        df["f1"] = df["f1"].fillna(0)

        df = df.sort_values(by="count", ascending=False)
        df = df[["count", "recall", "precision", "f1"]]
        return df

    @staticmethod
    def get_summary_plot(df):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # countの棒グラフとrecallの折れ線グラフを重ねて表示
        ax1 = axs[0]
        ax1.bar(df.index, df['count'], color='tab:blue', alpha=0.6, label='Count')
        ax1.set_ylabel('Count')
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['recall'], 'tab:orange', label='Recall')
        ax2.set_ylabel('Recall')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # countの棒グラフとprecisionの折れ線グラフを重ねて表示
        ax1 = axs[1]
        ax1.bar(df.index, df['count'], color='tab:blue', alpha=0.6, label='Count')
        ax1.set_ylabel('Count')
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['precision'], 'tab:green', label='Precision')
        ax2.set_ylabel('Precision')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # countの棒グラフとf1の折れ線グラフを重ねて表示
        ax1 = axs[2]
        ax1.bar(df.index, df['count'], color='tab:blue', alpha=0.6, label='Count')
        ax1.set_ylabel('Count')
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['f1'], 'tab:red', label='F1 Score')
        ax2.set_ylabel('F1 Score')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # X軸のラベルを非表示
        for ax in axs:
            ax.tick_params(axis='x', labelsize=0)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def get_low_metrics_brand(metrics: pd.DataFrame, count_threshold: int, metrics_threshold) -> pd.DataFrame:
        metrics = metrics[metrics["count"] >= count_threshold]
        f1_metrics = metrics[metrics["f1"] <= metrics_threshold]
        precision_metrics = metrics[metrics["precision"] <= metrics_threshold]
        recall_metrics = metrics[metrics["recall"] <= metrics_threshold]
        metrics = pd.concat([f1_metrics, precision_metrics, recall_metrics], axis=0).drop_duplicates()
        metrics = metrics.sort_values(by="count", ascending=False)
        return metrics

    def get_specified_brand_incorrect_samples(self, brand: str) -> pd.DataFrame:
        df = self.df[self.df["answer"] == brand]
        df = df[df["correct"] == 0]
        return df

    def get_specified_brand_correct_samples(self, brand: str) -> pd.DataFrame:
        df = self.df[self.df["answer"] == brand]
        df = df[df["correct"] == 1]
        return df


if __name__ == "__main__":
    path = "D:/datasets/phishing_identification/qa_results/qa_validation_result.csv"
    analyzer = ResultAnalyzer(path)
    print(analyzer.df.columns)

    print(f"accuracy : {analyzer.get_accuracy()}")
    metrics_df = analyzer.get_metrics_by_brand()
    print(metrics_df)
    metrics_df.to_csv("D:/datasets/phishing_identification/qa_results/qa_validation_result_metrics.csv", index=False)
    analyzer.get_summary_plot(metrics_df)
    low_metrics_df = analyzer.get_low_metrics_brand(metrics_df, 10, 0.8)
    print(low_metrics_df)

    print("")
    print("### incorrect sample ###")
    incorrect_df = analyzer.get_specified_brand_incorrect_samples("Alaska USA Federal Credit Union")
    print(f"inference : {incorrect_df['inference'].tolist()[0]}")
    print(f"identified : {incorrect_df['identified'].tolist()[0]}")
    with open("D:/datasets/phishing_identification/qa_results/incorrect_sample.txt", "w", encoding="utf-8") as f:
        f.write(incorrect_df["html"].tolist()[0])
    print("")
    print("### correct sample ###")
    correct_df = analyzer.get_specified_brand_correct_samples("Alaska USA Federal Credit Union")
    print(f"inference : {correct_df['inference'].tolist()[0]}")
    print(f"identified : {correct_df['identified'].tolist()[0]}")
    with open("D:/datasets/phishing_identification/qa_results/correct_sample.txt", "w", encoding="utf-8") as f:
        f.write(correct_df["html"].tolist()[0])
