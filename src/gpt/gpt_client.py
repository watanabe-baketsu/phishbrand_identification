import os
import time
from os.path import join, dirname

import pandas as pd
from openai import OpenAI, APIError
from dotenv import load_dotenv
from datasets import load_from_disk

from src.gpt.prompt import system_prompt


load_dotenv()
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


class GPTClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _request_gpt(self, html_code: str, model_name: str = "gpt-3.5-turbo-1106"):
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": html_code}
            ],
            temperature=1.0,
            max_tokens=256,
        )
        return response

    def request_manager(self, html_code: str, model_name: str = "gpt-3.5-turbo-1106") -> str:
        try:
            response = self._request_gpt(html_code, model_name)
            inference_brand = response.choices[0].message.content.strip()
            return inference_brand
        except APIError as api_error:
            if api_error.code == 429:
                print("Too many requests, sleep 60 seconds...")
                time.sleep(60)
                print("Retry request...")
                response = self._request_gpt(html_code, model_name)
                inference_brand = response.choices[0].message.content.strip()
                return inference_brand
            elif api_error.code == 400:
                print("The request is too large")
                html_code = html_code[:4*10000]
                print("Retry request...")
                response = self._request_gpt(html_code, model_name)
                inference_brand = response.choices[0].message.content.strip()
                return inference_brand
        except Exception as error:
            raise error


def load_checkpoint_df(checkpoint_path: str) -> pd.DataFrame:
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)
    else:
        df = dataset.to_pandas()
    return df


if __name__ == "__main__":
    model = "gpt-3.5-turbo-1106"  # "gpt-4-1106-preview"
    gpt_client = GPTClient()

    validation_length = 4000
    # load dataset
    base_path = "D:/datasets/phishing_identification"
    dataset = load_from_disk(f"{base_path}/phish-html-en-qa").select(range(10000, 10000 + validation_length))

    # for testing
    # brand = request_gpt(dataset[0]["context"], model)
    # print(brand)

    checkpoint_df = load_checkpoint_df(f"{base_path}/gpt_results/{model}-result.csv")
    # add column "inference" if not exists
    if "inference" not in checkpoint_df.columns:
        checkpoint_df["inference"] = ""
    # select samples that "inference" column is empty
    targets = checkpoint_df[checkpoint_df["inference"] == ""]
    count = 0
    for _, sample in targets.iterrows():
        try:
            inference = gpt_client.request_manager(sample["context"], model)
            checkpoint_df.loc[checkpoint_df["id"] == sample["id"], "inference"] = inference
        except Exception as e:
            print(e)
            checkpoint_df.to_csv(f"{base_path}/{model}-result.csv", index=False)

        time.sleep(5)
        count += 1
        print(f"\r{count} / {len(targets)}", end="")

    checkpoint_df.to_csv(f"{base_path}/gpt_results/{model}-result.csv", index=False)
    print(f"done : {len(checkpoint_df[checkpoint_df['inference'] != ''])}")
