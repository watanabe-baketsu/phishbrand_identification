import chardet
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect
import glob


class RawDatasetAnalysis:
    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def find_encoding(file):
        rawdata = open(file, 'rb').read()
        result = chardet.detect(rawdata)
        return result['encoding']

    def count_chars_and_stats(self):
        files = glob.glob(self.path + '/**/html.txt', recursive=True)
        file_lengths = []

        for file in files:
            encoding = self.find_encoding(file)
            with open(file, 'r', encoding=encoding, errors="ignore") as f:
                content = f.read()
                if len(content) > 0:
                    file_lengths.append(len(content))

        # print basic statistics
        print('Max:', max(file_lengths))
        print('Min:', min(file_lengths))
        print('Avg:', sum(file_lengths) / len(file_lengths))

        # plot distribution
        sns.displot(file_lengths, kde=True)
        plt.xlabel('File lengths')
        plt.ylabel('Count')
        plt.show()

    def lang_count_and_stats(self):
        files = glob.glob(self.path + '/**/html.txt', recursive=True)
        lang_counts = {}

        for file in files:
            encoding = self.find_encoding(file)
            with open(file, 'r', encoding=encoding) as f:
                content = f.read()
                lang = detect(content)
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

        print(lang_counts)

        # plot
        keys = list(lang_counts.keys())
        values = list(lang_counts.values())
        plt.bar(keys, values)
        plt.xlabel('Languages')
        plt.ylabel('File count')
        plt.show()


def analyze_char_stats(path: str):
    analysis = RawDatasetAnalysis(path)
    analysis.count_chars_and_stats()


def analyze_lang_stats(path: str):
    analysis = RawDatasetAnalysis(path)
    analysis.lang_count_and_stats()


def main():
    path = 'D:/datasets/phishpedia_vanilla/phish_sample_30k'
    analyze_char_stats(path)
    analyze_lang_stats(path)


if __name__ == '__main__':
    main()
