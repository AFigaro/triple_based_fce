import json
import os
import random
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

###########################################
# CONFIGURATION – EDIT THESE AS NEEDED
###########################################
model_name = 'gpt4o_mini'
# We assume the experiment directory is fixed:
experiment_dir = os.path.join("triplet_extraction_results", model_name, "experiments_single_task_topk", "experiment1")
# Input file with candidate responses
input_file = os.path.join(experiment_dir, "triplets_single_task.json")

# The six evaluation modes we want to compute together:
evaluation_modes = [
    "average_comprehensive",  # Cartesian product averaging
    "average_pairwise",       # Aligned pair averaging
    "comprehensive_random",   # Random selection from all pairs
    "comprehensive_top_k",    # Best match (all pairs) closest to label
    "pairwise_random",        # Random selection among aligned pairs
    "pairwise_top_k"          # Best match (aligned pairs) closest to label
]

# If you wish to use a gold standard for filtering, set the frank_file path.
# (For these combined modes we assume filtering by Frank IDs is desired.)
frank_file = r"D:\Repos\LLMs_FCE_Paper\data\frank_test.json"
use_frank_filter = True  # Set to False to process all entries regardless

# Output: All results files will be saved in experiment_dir,
# and a combined statistics file "statistics_all.txt" will be written there.
###########################################

###########################################
# HELPER FUNCTIONS
###########################################
def remove_brackets(s):
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
    Given two strings (each with newline‐separated triplets),
    compute the ratio of summary triplets that appear in the article.
    """
    summary_triplets = [remove_brackets(line) for line in summary_str.split("\n") if line.strip()]
    article_triplets = [remove_brackets(line) for line in article_str.split("\n") if line.strip()]
    total = len(summary_triplets)
    if total == 0:
        return 0
    common = sum(1 for triplet in summary_triplets if triplet in article_triplets)
    return common / total

def process_entry(entry, mode):
    """
    Process one response entry according to the given evaluation mode.
    Returns a dictionary with at least keys: "id", "label", "ratio".
    Depending on the mode, it may also return "article" and "summary" (the candidate chosen).
    
    Assumes that for these modes the entry has candidate lists under keys "articles" and "summaries".
    """
    entry_id = entry.get("id")
    label = convert_label(entry.get("label"))
    if label is None:
        return None
    result = {"id": entry_id, "label": label}
    
    # For all these combined modes we assume top_k responses (i.e. candidate lists):
    articles = entry.get("articles", [])
    summaries = entry.get("summaries", [])
    
    if mode == "average_comprehensive":
        # Cartesian product: consider all (article, summary) pairs.
        candidate_ratios = []
        for art in articles:
            for summ in summaries:
                candidate_ratios.append(compute_ratio(art, summ))
        avg_ratio = sum(candidate_ratios) / len(candidate_ratios) if candidate_ratios else 0
        result["ratio"] = avg_ratio

    elif mode == "average_pairwise":
        # Assume the candidate lists are aligned; process pairwise up to the minimum length.
        n_candidates = min(len(articles), len(summaries))
        candidate_ratios = []
        for i in range(n_candidates):
            candidate_ratios.append(compute_ratio(articles[i], summaries[i]))
        avg_ratio = sum(candidate_ratios) / n_candidates if n_candidates > 0 else 0
        result["ratio"] = avg_ratio

    elif mode == "comprehensive_random":
        # From the full cartesian product, randomly select one candidate pair.
        candidate_pairs = [(art, summ) for art in articles for summ in summaries]
        if candidate_pairs:
            art, summ = random.choice(candidate_pairs)
            result["article"] = art
            result["summary"] = summ
            result["ratio"] = compute_ratio(art, summ)
        else:
            result.update({"article": "", "summary": "", "ratio": 0})

    elif mode == "comprehensive_top_k":
        # From all candidate pairs, choose the one with ratio closest to the label.
        best_ratio = None
        best_art = ""
        best_summ = ""
        for art in articles:
            for summ in summaries:
                r = compute_ratio(art, summ)
                if best_ratio is None or abs(r - label) < abs(best_ratio - label):
                    best_ratio = r
                    best_art = art
                    best_summ = summ
        result.update({
            "article": best_art,
            "summary": best_summ,
            "ratio": best_ratio if best_ratio is not None else 0
        })

    elif mode == "pairwise_random":
        # Assume aligned candidate lists; randomly select one index.
        n_candidates = min(len(articles), len(summaries))
        if n_candidates > 0:
            i = random.randint(0, n_candidates - 1)
            art = articles[i]
            summ = summaries[i]
            result["article"] = art
            result["summary"] = summ
            result["ratio"] = compute_ratio(art, summ)
        else:
            result.update({"article": "", "summary": "", "ratio": 0})

    elif mode == "pairwise_top_k":
        # Among aligned candidate pairs, choose the pair with ratio closest to the label.
        n_candidates = min(len(articles), len(summaries))
        best_ratio = None
        best_art = ""
        best_summ = ""
        for i in range(n_candidates):
            r = compute_ratio(articles[i], summaries[i])
            if best_ratio is None or abs(r - label) < abs(best_ratio - label):
                best_ratio = r
                best_art = articles[i]
                best_summ = summaries[i]
        result.update({
            "article": best_art,
            "summary": best_summ,
            "ratio": best_ratio if best_ratio is not None else 0
        })
    else:
        # Unknown mode.
        return None
    return result

###########################################
# MAIN PIPELINE FOR COMBINED EVALUATION
###########################################
def main():
    # Check that the input file exists.
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    # Load responses from the input file.
    with open(input_file, "r", encoding="utf-8") as f:
        responses = json.load(f)

    # If filtering by Frank IDs is desired, load the frank file.
    frank_ids = set()
    if use_frank_filter:
        if not os.path.exists(frank_file):
            print(f"Frank file not found: {frank_file}")
            return
        with open(frank_file, "r", encoding="utf-8") as f:
            frank_entries = json.load(f)
        frank_ids = {entry.get("id") for entry in frank_entries}

    # For each evaluation mode, process every response (optionally filtering by Frank IDs).
    all_results = {}   # key: mode, value: list of processed entries
    for mode in evaluation_modes:
        mode_results = []
        for entry in tqdm(responses, desc=f"Processing for mode {mode}"):
            entry_id = entry.get("id")
            if use_frank_filter and entry_id not in frank_ids:
                continue
            processed = process_entry(entry, mode)
            if processed is not None:
                mode_results.append(processed)
        all_results[mode] = mode_results

        # Save individual results for this mode.
        output_path = os.path.join(experiment_dir, f"results_{mode}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mode_results, f, indent=4)
        print(f"Saved results for mode '{mode}' to {output_path}")

    # Now compute correlation statistics for each mode and combine them into one file.
    stats_lines = []
    for mode, results in all_results.items():
        # Only consider entries with a numeric label.
        numeric_results = [res for res in results if isinstance(res.get("label"), (int, float))]
        if len(numeric_results) < 2:
            stats_lines.append(f"{mode}: Not enough numeric data to compute correlation.\n")
            continue
        labels = [res["label"] for res in numeric_results]
        ratios = [res["ratio"] for res in numeric_results]
        try:
            pearson_corr, p_val = pearsonr(labels, ratios)
            spearman_corr, sp_val = spearmanr(labels, ratios)
        except Exception as e:
            stats_lines.append(f"{mode}: Error computing correlation: {e}\n")
            continue
        non_zero = sum(1 for r in ratios if r > 0)
        total = len(numeric_results)
        stats_lines.append(
            f"{mode}:\n"
            f"  Pearson correlation: {pearson_corr:.4f} (p-value: {p_val:.4f})\n"
            f"  Spearman correlation: {spearman_corr:.4f} (p-value: {sp_val:.4f})\n"
            f"  Non-zero ratios: {non_zero} out of {total}\n"
        )
    stats_text = "\n".join(stats_lines)
    stats_file = os.path.join(experiment_dir, "statistics_all.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(stats_text)
    print(f"Combined statistics saved to {stats_file}")

if __name__ == "__main__":
    main()
