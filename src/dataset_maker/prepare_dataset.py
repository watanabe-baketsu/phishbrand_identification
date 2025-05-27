import ast
import base64
import json
import os
import re
from typing import Literal

from bs4 import BeautifulSoup
from langdetect import detect

from datasets import Dataset
from src.config import (
    BENIGN_SAMPLE_30K,
    PHISH_FULL,
    PHISH_HTML_EN,
    PHISH_HTML_PICKUP_EN,
    PHISH_TEXT_EN,
    TRAINING_JSONL,
)


def replace_multiple_newlines(text):
    return re.sub(r"\n+", "\n", text)


class DatasetGenerator:
    def __init__(self, base_path: str, label: Literal["phish", "benign"]):
        self.base_path = base_path
        self.label = label

    def _read_html_json_pairs(self):
        html_json_pairs = []
        for target_dir, _, _ in os.walk(self.base_path):
            # Only append if the html and info files exist.
            if os.path.exists(target_dir + "/html.txt") and os.path.exists(
                target_dir + "/info.txt"
            ):
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
        pattern = r"^[A-Za-z0-9+/]*={0,2}$"
        if re.match(pattern, text):
            try:
                base64.b64decode(text)
                return True
            except:
                return False
        return False

    @staticmethod
    def _get_only_text(html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text()

    @staticmethod
    def _is_english(text: str) -> bool:
        lang = detect(text)
        if lang == "en":
            return True
        else:
            return False

    def _remove_base64(self, soup: BeautifulSoup) -> str:
        # Remove img tags with base64 encoded src attributes
        for img_tag in soup.find_all("img", src=True):
            src_value = img_tag["src"]
            if src_value.startswith("data:image") and "base64" in src_value:
                # Assume base64 encoding if it starts with 'data:image'
                img_tag.decompose()

        text = str(soup)
        for text_node in soup.stripped_strings:
            if self._is_base64(text_node):
                text = text.replace(text_node, "")

        return text

    def _shorten_html(self, html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")
        allowed_tags = [
            "head",
            "title",
            "body",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "strong",
            "a",
            "img",
            "hr",
            "table",
            "tbody",
            "tr",
            "th",
            "td",
            "ol",
            "ul",
            "li",
            "ruby",
            "label",
        ]

        for tag in soup.find_all(True):
            if tag.name not in allowed_tags:
                if not tag.find_all(True, recursive=False):
                    tag.decompose()

        text = self._remove_base64(soup)

        return text

    def _shorten_by_text_html(self, html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")
        allowed_tags = [
            "head",
            "title",
            "body",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "strong",
            "a",
            "img",
            "hr",
            "table",
            "tbody",
            "tr",
            "th",
            "td",
            "ol",
            "ul",
            "li",
            "ruby",
            "label",
        ]

        for tag in soup.find_all(True):
            if tag.name not in allowed_tags:
                if not tag.find_all(True, recursive=False):
                    tag.decompose()
            if not tag.get_text(strip=True):
                tag.decompose()

        text = self._remove_base64(soup)

        return text

    def generate_shortened_html_dataset(self) -> Dataset:
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
                text = replace_multiple_newlines(self._get_only_text(html))
                if self._is_english(html):  # only english html
                    info = ast.literal_eval(info)
                    html = self._shorten_html(html)
                    data = {
                        "html": f"<p>{text}</p>{html}",
                        "brand": info.get("brand"),
                        "url": info.get("url"),
                        "host": info.get("host"),
                        "label": self.label,
                    }
                    dataset.append(data)
            except Exception as e:
                print(e)
        dataset = Dataset.from_dict({self.label: Dataset.from_list(dataset)})

        dataset.save_to_disk(PHISH_HTML_EN)

        return dataset

    def generate_pickup_html_dataset(self) -> Dataset:
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
                if self._is_english(html):  # only english html
                    info = ast.literal_eval(info)
                    data = {
                        "html": self._shorten_by_text_html(html),
                        "brand": info.get("brand"),
                        "url": info.get("url"),
                        "host": info.get("host"),
                        "label": self.label,
                    }
                    dataset.append(data)
            except Exception as e:
                print(e)
        dataset = Dataset.from_dict({self.label: Dataset.from_list(dataset)})

        dataset.save_to_disk(PHISH_HTML_PICKUP_EN)

        return dataset

    def generate_text_only_dataset(self) -> Dataset:
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
                text = self._get_only_text(html)
                if self._is_english(text) and text != "":  # onlu english html
                    info = ast.literal_eval(info)
                    data = {
                        "text": replace_multiple_newlines(text),
                        "brand": info.get("brand"),
                        "url": info.get("url"),
                        "host": info.get("host"),
                        "label": self.label,
                    }
                    dataset.append(data)
            except Exception as e:
                print(e)
        dataset = Dataset.from_dict({self.label: Dataset.from_list(dataset)})

        dataset.save_to_disk(PHISH_TEXT_EN)

        return dataset

    def generate_full_dataset(self):
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
                text = self._get_only_text(html)
                if text != "":  # only english html
                    info = ast.literal_eval(info)
                    data = {
                        "text": replace_multiple_newlines(text),
                        "html": html,
                        "brand": info.get("brand"),
                        "url": info.get("url"),
                        "host": info.get("host"),
                        "label": self.label,
                    }
                    dataset.append(data)
            except Exception as e:
                print(e)
        dataset = Dataset.from_dict({self.label: Dataset.from_list(dataset)})

        dataset.save_to_disk(PHISH_FULL)

        return dataset

    def generate_summarization_training_dataset(self):
        html_info_pairs = self._read_html_json_pairs()
        dataset = []

        for pair in html_info_pairs:
            html_path = pair["html_path"]
            info_path = pair["info_path"]
            try:
                with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
                with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
                    info = f.read()
                text = self._get_only_text(html)
                host = info.split("//")[-1].split(".")[:-1]
                if text != "" and self._is_english(text):  # only english html
                    data = {"text": replace_multiple_newlines(text), "target": host}
                    dataset.append(data)
            except Exception as e:
                print(e)
        with open(
            TRAINING_JSONL,
            "w",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            for data in dataset:
                json.dump(data, f)
                f.write("\n")
        print("done")


if __name__ == "__main__":

    phish_base_path = BENIGN_SAMPLE_30K
    phish_label = "benign"
    dataset_generator = DatasetGenerator(phish_base_path, phish_label)
    dataset_generator.generate_summarization_training_dataset()
