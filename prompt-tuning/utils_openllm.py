import json
import openai
from pydantic import BaseModel
from typing import List
from langchain.output_parsers import PydanticOutputParser



def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading {file_path} as JSON array: {e}")
                data = []
    return data


def save_responses(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


class Triplet(BaseModel):
    subject: str
    predicate: str
    obj: str

class StructuredOutput(BaseModel):
    article_triplets: List[Triplet]
    summary: str
    summary_triplets: List[Triplet]

    def formatted_article(self) -> str:
        # Format only the article triplets
        article_str = "\n".join(
            f"({t.subject}, {t.predicate}, {t.obj})" for t in self.article_triplets
        )
        return f"[ {article_str} ]"

    def formatted_summary(self) -> str:
        # Format only the summary triplets (no extra text)
        summary_triplets_str = "\n".join(
            f"({t.subject}, {t.predicate}, {t.obj})" for t in self.summary_triplets
        )
        return f"[ {summary_triplets_str} ]"