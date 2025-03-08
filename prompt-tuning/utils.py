import json
import openai
import ollama
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


def call_model(top_k, developer_prompt, combined_prompt, model_name, entry_id=None):
    """
    Calls either the OpenAI ChatCompletion API or Ollama's chat API based on the model_name,
    then processes the response by splitting the returned text on "<summary>" into article and summary parts.
    
    Parameters:
      top_k (bool or int): 
         - For OpenAI: if truthy (e.g., an integer k), the 'n' parameter is set to k to request multiple completions.
         - For Ollama: if truthy, the function makes k separate calls in a loop.
         - If False, a single call is made.
      developer_prompt (str): The developer (or system) prompt.
      combined_prompt (str): The user prompt.
      model_name (str): The model name. If it starts with "gpt", an OpenAI call is made; otherwise, an Ollama call is made.
      entry_id (optional): Identifier used for logging error messages.
      
    Returns:
      For top_k true:
        A dictionary with:
          - "articles": list of article outputs.
          - "summaries": list of summary outputs.
      For top_k false:
        A dictionary with:
          - "article": article output.
          - "summary": summary output.
    """
    if model_name.startswith("gpt"):
        messages = [
            {'role': 'developer', 'content': developer_prompt},
            {'role': 'user', 'content': combined_prompt}
        ]
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        if top_k:
            params["n"] = top_k
        try:
            response = openai.ChatCompletion.create(**params)
        except Exception as e:
            print(f"API call error for id {entry_id}: {e}")
            return None
        if top_k:
            outputs = [choice["message"]["content"] for choice in response["choices"]]
            return outputs
        else:
            return response["choices"][0]["message"]["content"]
    else:
        messages = [
            {'role': 'system', 'content': developer_prompt},
            {'role': 'user', 'content': combined_prompt}
        ]
        if top_k:
            responses = []
            for i in range(top_k):
                response = ollama.chat(model=model_name, messages=messages, format=StructuredOutput.model_json_schema())
                responses.append(response["message"]["content"])
            return responses
        else:
            response = ollama.chat(model=model_name, messages=messages, format=StructuredOutput.model_json_schema())
            return response["message"]["content"]


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