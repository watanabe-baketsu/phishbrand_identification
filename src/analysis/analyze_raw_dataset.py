import ast
import glob
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langdetect import LangDetectException, detect
from src.config import PHISHPEDIA_VANILLA


class RawDatasetAnalysis:
    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def find_encoding(file):
        rawdata = open(file, "rb").read()
        result = chardet.detect(rawdata)
        return result["encoding"]

    def count_chars_and_stats(self):
        files = glob.glob(self.path + "/**/html.txt", recursive=True)
        file_lengths = []

        for file in files:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if len(content) > 0:
                    file_lengths.append(len(content))

        # print basic statistics
        print("Max:", max(file_lengths))
        print("Min:", min(file_lengths))
        print("Avg:", sum(file_lengths) / len(file_lengths))
        print("Median:", np.median(file_lengths))
        print("Std:", np.std(file_lengths))
        print("Q1:", np.quantile(file_lengths, 0.25))
        print("Q2:", np.quantile(file_lengths, 0.5))
        print("Q3:", np.quantile(file_lengths, 0.75))
        # plot distribution
        # bar plot
        range_size = 20000
        counts = Counter((x // range_size) * range_size for x in file_lengths)

        sorted_counts = sorted(counts.items())
        ranges = [str(x[0]) + "-" + str(x[0] + range_size) for x in sorted_counts]
        counts = [x[1] for x in sorted_counts]

        fig, ax1 = plt.subplots()  # グラフのサイズを調整
        ax1.bar(ranges, counts, color="b", alpha=0.6, label="Counts")
        ax1.set_ylabel("Counts", color="b", fontsize=8)
        ax1.tick_params(
            "y", colors="b", labelsize=8
        )  # y軸の目盛りサイズをフォントサイズ8に設定
        ax1.set_xticks(range(len(ranges)))
        ax1.set_xticklabels(ranges, rotation=-90, ha="left", fontsize=6)

        # 累積グラフ（線グラフ）
        sum_counts = []
        sum_count = 0
        for count in counts:
            sum_count += count
            sum_counts.append(sum_count)
        # convert list to pandas series
        sum_counts = pd.Series(sum_counts)

        ax2 = ax1.twinx()
        ax2.plot(ranges, sum_counts, color="r", marker="o", label="Sum")
        ax2.set_ylabel("Sum", color="r", fontsize=8)
        ax2.tick_params(
            "y", colors="r", labelsize=8
        )  # y軸の目盛りサイズをフォントサイズ8に設定
        ax2.set_ylim(bottom=0)  # 折れ線グラフのメモリを0スタートにする

        plt.tight_layout()
        plt.savefig("file_lengths_distribution.pdf", bbox_inches="tight")

    def lang_count_and_stats(self):
        files = glob.glob(self.path + "/**/html.txt", recursive=True)
        lang_counts = {}

        for file in files:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if len(content) >= 20:
                    try:
                        lang = detect(content)
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
                    except LangDetectException as e:
                        print(f"Language detection failed for file {file}: {e}")
                else:
                    print(f"Skipping file {file} due to insufficient text length.")

        print(lang_counts)

        # plot
        keys = list(lang_counts.keys())
        values = list(lang_counts.values())
        bars = plt.bar(keys, values)
        plt.xlabel("Languages")
        plt.ylabel("File count")
        # 各棒グラフの上にサンプル数を表示
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.tight_layout()
        plt.savefig("language_distribution.pdf")

    def brand_count_and_stats(self):
        files = glob.glob(self.path + "/**/info.txt", recursive=True)
        brand_counts = {}

        for file in files:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                try:
                    info = f.read()
                    info = info.replace("'", '"')
                    info = ast.literal_eval(info)
                    brand = info.get("brand", "unknown")
                    brand_counts[brand] = brand_counts.get(brand, 0) + 1
                except Exception as e:
                    print(f"Error while reading file {file}: {e}")

        # サンプル数の多い順にブランドを並べ替え
        sorted_brand_counts = sorted(
            brand_counts.items(), key=lambda x: x[1], reverse=True
        )
        keys = [item[0] for item in sorted_brand_counts]
        values = [item[1] for item in sorted_brand_counts]

        # plot
        fig, ax1 = plt.subplots()
        ax1.bar(keys, values, color="b", label="Count")
        ax1.set_ylabel("File count")
        ax1.tick_params("y", colors="b")
        ax1.set_xticks([])  # X軸のブランド名を表示しない

        # 累積グラフ（線グラフ）
        sum_counts = [
            sum(values[: i + 1]) for i in range(len(values))
        ]  # 累積サンプル数を計算
        ax2 = ax1.twinx()
        ax2.plot(keys, sum_counts, color="r", marker="o", label="Sum")
        ax2.set_ylabel("Sum", color="r")
        ax2.tick_params("y", colors="r")
        ax2.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig("brand_distribution.pdf")

        print(f"Total number of brands: {len(keys)}")


def analyze_char_stats(path: str):
    analysis = RawDatasetAnalysis(path)
    analysis.count_chars_and_stats()


def analyze_lang_stats(path: str):
    analysis = RawDatasetAnalysis(path)
    analysis.lang_count_and_stats()


def analyze_brand_stats(path: str):
    analysis = RawDatasetAnalysis(path)
    analysis.brand_count_and_stats()


def main():
    path = PHISHPEDIA_VANILLA
    # ProcessPoolExecutorを使用して並列処理を行う
    with ProcessPoolExecutor() as executor:
        # 各関数を並列に実行するためのFutureオブジェクトを取得
        executor.submit(analyze_char_stats, path)
        # executor.submit(analyze_lang_stats, path)
        # executor.submit(analyze_brand_stats, path)
        pass


if __name__ == "__main__":
    main()
