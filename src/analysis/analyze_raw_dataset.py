import ast
import glob
import os
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
        csv_file = "file_lengths_stats.csv"

        # Load from existing CSV file if it exists
        if os.path.exists(csv_file):
            print("Loading statistical data from existing CSV file...")
            df = pd.read_csv(csv_file)

            # Display statistical information
            stats = df.iloc[0]
            print("Max:", stats["max"])
            print("Min:", stats["min"])
            print("Avg:", stats["avg"])
            print("Median:", stats["median"])
            print("Std:", stats["std"])
            print("Q1:", stats["q1"])
            print("Q2:", stats["q2"])
            print("Q3:", stats["q3"])

            # Load distribution data
            distribution_csv = "file_lengths_distribution.csv"
            if os.path.exists(distribution_csv):
                dist_df = pd.read_csv(distribution_csv)
                ranges = dist_df["range"].tolist()
                counts = dist_df["count"].tolist()
                sum_counts = dist_df["cumulative_count"].tolist()
            else:
                print("Distribution data not found. Calculating new data...")
                ranges, counts, sum_counts = self._calculate_distribution()
        else:
            print("Calculating new statistical data...")
            files = glob.glob(self.path + "/**/html.txt", recursive=True)
            file_lengths = []

            for file in files:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if len(content) > 0:
                        file_lengths.append(len(content))

            # Calculate statistical information
            max_val = max(file_lengths)
            min_val = min(file_lengths)
            avg_val = sum(file_lengths) / len(file_lengths)
            median_val = np.median(file_lengths)
            std_val = np.std(file_lengths)
            q1_val = np.quantile(file_lengths, 0.25)
            q2_val = np.quantile(file_lengths, 0.5)
            q3_val = np.quantile(file_lengths, 0.75)

            # Display statistical information
            print("Max:", max_val)
            print("Min:", min_val)
            print("Avg:", avg_val)
            print("Median:", median_val)
            print("Std:", std_val)
            print("Q1:", q1_val)
            print("Q2:", q2_val)
            print("Q3:", q3_val)

            # Save statistical information to CSV
            stats_df = pd.DataFrame(
                {
                    "max": [max_val],
                    "min": [min_val],
                    "avg": [avg_val],
                    "median": [median_val],
                    "std": [std_val],
                    "q1": [q1_val],
                    "q2": [q2_val],
                    "q3": [q3_val],
                    "total_files": [len(file_lengths)],
                }
            )
            stats_df.to_csv(csv_file, index=False)
            print(f"Statistical data saved to {csv_file}.")

            # Calculate distribution data
            ranges, counts, sum_counts = self._calculate_distribution_from_lengths(
                file_lengths
            )

            # Save distribution data to CSV
            distribution_csv = "file_lengths_distribution.csv"
            dist_df = pd.DataFrame(
                {"range": ranges, "count": counts, "cumulative_count": sum_counts}
            )
            dist_df.to_csv(distribution_csv, index=False)
            print(f"Distribution data saved to {distribution_csv}.")

        # Draw graph
        self._plot_distribution(ranges, counts, sum_counts)

    def _calculate_distribution_from_lengths(self, file_lengths):
        """Calculate distribution data from list of file lengths"""
        range_size = 20000
        counts = Counter((x // range_size) * range_size for x in file_lengths)

        sorted_counts = sorted(counts.items())
        ranges = [str(x[0]) + "-" + str(x[0] + range_size) for x in sorted_counts]
        counts = [x[1] for x in sorted_counts]

        # Calculate cumulative counts
        sum_counts = []
        sum_count = 0
        for count in counts:
            sum_count += count
            sum_counts.append(sum_count)

        return ranges, counts, sum_counts

    def _calculate_distribution(self):
        """Recalculate distribution data from existing files (fallback for incomplete CSV)"""
        files = glob.glob(self.path + "/**/html.txt", recursive=True)
        file_lengths = []

        for file in files:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if len(content) > 0:
                    file_lengths.append(len(content))

        return self._calculate_distribution_from_lengths(file_lengths)

    def _plot_distribution(self, ranges, counts, sum_counts):
        """Draw distribution graph"""
        fig, ax1 = plt.subplots(figsize=(12, 8))  # Make graph size larger

        # Set X-axis positions numerically
        x_positions = range(len(ranges))

        ax1.bar(x_positions, counts, color="b", alpha=0.6, label="Counts")
        ax1.set_ylabel("Counts", color="b", fontsize=12)
        ax1.tick_params("y", colors="b", labelsize=10)

        # Display X-axis labels with thinning (while placing at correct positions)
        step = max(1, len(ranges) // 30)  # Display approximately 10 labels

        # Set all positions but thin out the labels
        ax1.set_xticks(x_positions)
        display_labels = []
        for i, label in enumerate(ranges):
            if i % step == 0:  # Remove condition to display the last label
                display_labels.append(label)
            else:
                display_labels.append("")  # Hide with empty string

        ax1.set_xticklabels(display_labels, rotation=-45, ha="left", fontsize=12)
        ax1.set_xlabel("File Length Range (characters)", fontsize=12)

        # Cumulative graph (line graph)
        ax2 = ax1.twinx()
        ax2.plot(
            x_positions,
            sum_counts,
            color="r",
            marker="o",
            markersize=4,
            label="Cumulative",
        )
        ax2.set_ylabel("Cumulative Count", color="r", fontsize=12)
        ax2.tick_params("y", colors="r", labelsize=10)
        ax2.set_ylim(bottom=0)

        # Add grid for better visibility
        ax1.grid(True, alpha=0.3, axis="y")

        # Add title
        plt.title("Distribution of HTML File Lengths", fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig("file_lengths_distribution.pdf", bbox_inches="tight", dpi=300)
        print("Graph saved to file_lengths_distribution.pdf.")

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
        # Display sample count above each bar graph
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

        # Sort brands in descending order of sample count
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
        ax1.set_xticks([])  # Do not display brand names on X-axis

        # Cumulative graph (line graph)
        sum_counts = [
            sum(values[: i + 1]) for i in range(len(values))
        ]  # Calculate cumulative sample count
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
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Get Future objects to execute each function in parallel
        executor.submit(analyze_char_stats, path)
        # executor.submit(analyze_lang_stats, path)
        # executor.submit(analyze_brand_stats, path)
        pass


if __name__ == "__main__":
    main()
