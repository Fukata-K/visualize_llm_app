import csv
import random
from pathlib import Path


def get_random_prompt(csv_path: str = "data/prompt_sample.csv") -> str:
    """
    CSV ファイルからランダムにプロンプトを1つ取得する関数.

    Args:
        csv_path (str): CSV ファイルのパス.

    Returns:
        str: ランダムに選択されたプロンプト文字列.
    """
    csv_file = Path(csv_path)

    prompts = []
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompts.append(row["prompt"].strip())

    return random.choice(prompts)
