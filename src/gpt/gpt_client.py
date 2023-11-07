import os
from os.path import join, dirname

from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_from_disk

from src.gpt.prompt import system_prompt


load_dotenv()
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def request_gpt(html_code: str, model_name: str = "gpt-3.5-turbo-1106") -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": html_code}
        ],
        temperature=1.0,
        max_tokens=256,
    )
    inference = response.choices[0].message.content.strip()

    return inference


if __name__ == "__main__":
    model = "gpt-4-1106-preview"

    validation_length = 4000
    # load dataset
    base_path = "D:/datasets/phishing_identification"
    dataset = load_from_disk(f"{base_path}/phish-html-en-qa").select(range(10000, 10000 + validation_length))

    # for testing
    brand = request_gpt(dataset[0]["context"], model)
    print(brand)

