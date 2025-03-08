import json
import os
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

###########################################
# CONFIGURATION – EDIT THESE AS NEEDED
###########################################
model_name = 'gpt4o_mini'
# Experiment directory is assumed to be fixed:
experiment_dir = os.path.join("triplet_extraction_results", model_name, "experiments_single_task_topk", "experiment1")
# Input file for ranked (or top-1) evaluation:
input_file = os.path.join(experiment_dir, "triplets_ranked.json")

# Output file names
results_file = os.path.join(experiment_dir, "results_ranked.json")
stats_file = os.path.join(experiment_dir, "statistics_ranked.txt")

###########################################
# HELPER FUNCTIONS
###########################################
def remove_brackets(s):
    """Remove surrounding square brackets from a string."""
    s = s.strip()
    if s.startswith('['):
        s = s[1:]
    if s.endswith(']'):
        s = s[:-1]
    return s.strip()

def convert_label(label_raw):
    """Convert label (e.g. 'CORRECT'/'INCORRECT') to a numeric value."""
    if isinstance(label_raw, str):
        if label_raw.upper() == "CORRECT":
            return 1
        elif label_raw.upper() == "INCORRECT":
            return 0
        try:
            return float(label_raw)
        except Exception:
            return None
    elif isinstance(label_raw, (int, float)):
        return label_raw
    return None

def compute_ratio(article_str, summary_str):
    """
    Given two strings (each containing newline-separated triplets),
    compute the ratio of summary triplets that appear in the article.
    """
    summary_triplets = [remove_brackets(line) for line in summary_str.split("\n") if line.strip()]
    article_triplets = [remove_brackets(line) for line in article_str.split("\n") if line.strip()]
    total = len(summary_triplets)
    if total == 0:
        return 0
    common = sum(1 for triplet in summary_triplets if triplet in article_triplets)
    return common / total

###########################################
# MAIN PIPELINE
###########################################
def main():
    # Check that the input file exists.
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    # Load the ranked responses.
    with open(input_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    results = []
    for entry in tqdm(entries, desc="Processing entries", total=len(entries)):
        entry_id = entry.get("id")
        label = convert_label(entry.get("label"))
        if label is None:
            continue

        # For ranked evaluation, use pre-ranked fields.
        summary_str = entry.get("best_summary_triplet", "")
        article_str = entry.get("best_article_triplet", "")
        
        ratio = compute_ratio(article_str, summary_str)
        
        results.append({
            "id": entry_id,
            "label": label,
            "article": article_str,
            "summary": summary_str,
            "ratio": ratio
        })

    # Save the detailed results.
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")

    # Compute correlation statistics.
    numeric_results = [res for res in results if isinstance(res.get("label"), (int, float))]
    if len(numeric_results) < 2:
        stats_text = "Not enough numeric data to compute correlation.\n"
        print(stats_text)
    else:
        labels = [res["label"] for res in numeric_results]
        ratios = [res["ratio"] for res in numeric_results]
        pearson_corr, p_val = pearsonr(labels, ratios)
        spearman_corr, sp_val = spearmanr(labels, ratios)
        non_zero = sum(1 for r in ratios if r > 0)
        total = len(numeric_results)
        stats_text = (
            f"Pearson correlation between label and ratio: {pearson_corr:.4f} (p-value: {p_val:.4f})\n"
            f"Spearman correlation between label and ratio: {spearman_corr:.4f} (p-value: {sp_val:.4f})\n"
            f"Total samples with non-zero ratio: {non_zero} out of {total}\n"
        )
        print(stats_text)

    # Save statistics to file.
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(stats_text)
    print(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
    main()
