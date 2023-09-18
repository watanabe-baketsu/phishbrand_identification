import ast
import base64
import os
import re
from typing import Literal

from bs4 import BeautifulSoup
from datasets import Dataset


class DatasetGenerator:
    def __init__(self, base_path: str, label: Literal["phish", "benign"]):
        self.base_path = base_path
        self.label = label

    def _read_html_json_pairs(self):
        html_json_pairs = []
        for target_dir, _, _ in os.walk(self.base_path):
            # Only append if the html and info files exist.
            if os.path.exists(target_dir + "/html.txt") and os.path.exists(target_dir + "/info.txt"):
                html_path = target_dir + "/html.txt"
                info_path = target_dir + "/info.txt"
                data = {
                    "html_path": html_path,
                    "info_path": info_path,
                }
                html_json_pairs.append(data)

        return html_json_pairs

    @staticmethod
    def _is_base64(text: str) -> bool:
        pattern = r'^[A-Za-z0-9+/]*={0,2}$'
        if re.match(pattern, text):
            try:
                base64.b64decode(text)
                return True
            except:
                return False
        return False

    def _shorten_html(self, html_text: str) -> str:
        soup = BeautifulSoup(html_text, 'html.parser')
        allowed_tags = [
            'head', 'title', 'meta', 'body', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'strong', 'a', 'img', 'hr', 'table', 'tbody', 'tr', 'th', 'td',
            'ol', 'ul', 'li', 'ruby', 'label'
        ]

        for tag in soup.find_all(True):
            if tag.name not in allowed_tags:
                if not tag.find_all(True, recursive=False):
                    tag.decompose()

        # Remove img tags with base64 encoded src attributes
        for img_tag in soup.find_all('img', src=True):
            src_value = img_tag['src']
            if src_value.startswith('data:image') and "base64" in src_value:
                # Assume base64 encoding if it starts with 'data:image'
                img_tag.decompose()

        text = str(soup)
        for text_node in soup.stripped_strings:
            if self._is_base64(text_node):
                text = text.replace(text_node, "")

        return text

    def generate_dataset(self) -> Dataset:
        html_json_pairs = self._read_html_json_pairs()
        dataset = []
        for pair in html_json_pairs:
            html_path = pair["html_path"]
            info_path = pair["info_path"]
            try:
                with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
                with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
                    info = f.read()
                info = ast.literal_eval(info)
                data = {
                    "html": self._shorten_html(html),
                    "brand": info.get("brand"),
                    "url": info.get("url"),
                    "host": info.get("host"),
                    "label": self.label,
                }
                dataset.append(data)
            except Exception as e:
                print(e)
        dataset = Dataset.from_dict({self.label: Dataset.from_list(dataset)})

        return dataset


if __name__ == "__main__":
    # benign_base_path = "D:/datasets/phishing_identification/benign_sample_30k"
    # benign_label = "benign"
    # dataset_generator = DatasetGenerator(benign_base_path, benign_label)
    # benign_dataset = dataset_generator.generate_dataset()
    # benign_dataset.save_to_disk("D:/datasets/phishing_identification/benign")
    # print(benign_dataset.num_rows)

    phish_base_path = "D:/datasets/phishing_identification/phish_sample_30k"
    phish_label = "phish"
    dataset_generator = DatasetGenerator(phish_base_path, phish_label)
    phish_dataset = dataset_generator.generate_dataset()
    phish_dataset.save_to_disk("D:/datasets/phishing_identification/phish")
    print(phish_dataset.num_rows)
