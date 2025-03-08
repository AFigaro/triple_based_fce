import json
import os
from tqdm import tqdm
from utils import load_json_file, save_responses, StructuredOutput

from langchain.output_parsers import PydanticOutputParser
from langchain.llms import OpenLLM

# Set arguments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
experiment_k = 1  # Modify this parameter as needed
top_k = True
k = 3  # Number of completions per prompt when top_k is True
model_name = 'mistral'
dataset = "FactCheck"

# Server configuration for OpenLLM
server_url = 'http://131.155.68.20:3000'
server_type = 'http'

# Create experiment directory
if not top_k:
    experiment_dir = os.path.join(
        BASE_DIR,
        "triplet_extraction_results",
        model_name,
        "experiments_single_top1",
        f"experiment{experiment_k}"
    )
else:
    experiment_dir = os.path.join(
        BASE_DIR,
        "triplet_extraction_results",
        model_name,
        "experiments_single_topk",
        f"experiment{experiment_k}"
    )
os.makedirs(experiment_dir, exist_ok=False)

############################
#   Prompt Preparation     #
############################

# Load the developer prompt (system-level prompt)
with open("prompt-tuning/prompts/instruction_developer.txt", "r", encoding="utf-8") as f:
    developer_prompt = f.read()

# Load the task instructions, format, and examples parts for the user prompt
with open("prompt-tuning/prompts/single_task/instruction_single_task.txt", "r", encoding="utf-8") as f:
    instruction_prompt = f.read()

with open("prompt-tuning/prompts/single_task/format_single_task.txt", "r", encoding="utf-8") as f:
    format_prompt = f.read()

with open("prompt-tuning/prompts/single_task/examples_single_task.txt", "r", encoding="utf-8") as f:
    examples_prompt = f.read()

combined_prompt_template = (
    f"{instruction_prompt}\n"
    f"{format_prompt}\n"
    f"{examples_prompt}\n"
    "Article:\n{article}\n"
    "Summary:\n{summary}\n"
    "Triplets:\n"
)

# Create a LangChain output parser based on our StructuredOutput model
output_parser = PydanticOutputParser(pydantic_object=StructuredOutput)
format_instructions = output_parser.get_format_instructions()

#################################################
#   Defining Outputs in Experiment Directory    #
#################################################

# Save the combined prompt template for reference
combined_prompt_path = os.path.join(experiment_dir, "combined_prompt.txt")
with open(combined_prompt_path, "w", encoding="utf-8") as f:
    f.write(combined_prompt_template)

# Define checkpoint file in the experiment directory
checkpoint_file = os.path.join(experiment_dir, "triplets_single_task.json")
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        try:
            output_triplets = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading checkpoint file: {e}")
            output_triplets = []
else:
    output_triplets = []

# Create a set of already processed ids
processed_ids = set(item.get("id") for item in output_triplets)

########################
#   Reading Dataset    #
########################

if dataset == "FactCheck":
    data_files = ["data/factcheck_train.json", "data/frank_test.json"]
elif dataset == "FRANK":
    data_files = ["data/frank_test.json"]
elif dataset == "FactCheck-train":
    data_files = ["data/factcheck_train.json"]

# Calculate total entries (optional, for progress display)
total_entries = 0
for file_path in data_files:
    if os.path.exists(file_path):
        total_entries += len(load_json_file(file_path))

current_count = len(processed_ids)

###################################
#   Main Loop with OpenLLM Calls  #
###################################

# Initialize OpenLLM client once
llm = OpenLLM(server_url=server_url, server_type=server_type)

for file_path in data_files:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    data_entries = load_json_file(file_path)
    for entry in tqdm(data_entries, desc=f"Processing {file_path}", unit="entry"):
        entry_id = entry.get("id")
        if entry_id in processed_ids:
            continue

        result = {"id": entry_id}

        # Build combined prompt by inserting article and summary into the template
        article_text = entry.get("article", "")
        summary_text = entry.get("summary", "")
        combined_prompt = combined_prompt_template.format(article=article_text, summary=summary_text)
        
        # Combine the developer prompt and the combined prompt
        final_prompt = developer_prompt + "\n" + combined_prompt

        if top_k:
            article_outputs = []
            summary_outputs = []
            # Query the model k times
            for i in range(k):
                response_data = llm(final_prompt)
                if response_data is None:
                    continue
                try:
                    structured_obj = output_parser.parse(response_data)
                    article_outputs.append(structured_obj.formatted_article())
                    summary_outputs.append(structured_obj.formatted_summary())
                except Exception as e:
                    print(f"Error parsing structured output for id {entry_id} in iteration {i}: {e}")
                    continue
            result["articles"] = article_outputs
            result["summaries"] = summary_outputs
        else:
            response_data = llm(final_prompt)
            if response_data is None:
                continue
            try:
                structured_obj = output_parser.parse(response_data)
                result["articles"] = structured_obj.formatted_article()
                result["summaries"] = structured_obj.formatted_summary()
            except Exception as e:
                print(f"Error parsing structured output for id {entry_id}: {e}")
                continue

        # Convert label: "INCORRECT" -> 0, "CORRECT" -> 1.
        label_str = entry.get("label", "").strip().upper()
        if label_str == "INCORRECT":
            result["label"] = 0
        elif label_str == "CORRECT":
            result["label"] = 1
        else:
            result["label"] = entry.get("label")

        output_triplets.append(result)
        processed_ids.add(entry_id)
        current_count += 1

        # Incremental saving after each entry.
        save_responses(output_triplets, checkpoint_file)
        tqdm.write(f"Processed {current_count} / {total_entries} entries.")

print(f"Total entries processed: {current_count} / {total_entries}")
print("Processing complete. Results saved to", checkpoint_file)
