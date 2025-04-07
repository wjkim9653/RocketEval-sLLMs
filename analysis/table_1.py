import os
import json
from collections import defaultdict

# Load GPT-4o scores first
def load_gpt4o_scores():
    scores = {}
    gpt4o_dir = os.path.expanduser("data/mt-bench/score/gpt-4o")
    for filename in os.listdir(gpt4o_dir):
        if filename.endswith(".json"):
            model_test = filename[:-5]  # Remove .json
            with open(os.path.join(gpt4o_dir, filename)) as f:
                data = json.load(f)
                for entry in data:
                    key = (model_test, entry["session_id"])
                    scores[key] = entry["score"]
    return scores

# Calculate agreement percentages
def calculate_agreement(gpt4o_scores):
    results = defaultdict(lambda: defaultdict(lambda: {"total": 0, "agreed": 0}))
    root_dir = os.path.expanduser("data/mt-bench/score")
    
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_path) or dir_name == "gpt-4o":
            continue
        
        # Parse judge model and checklist type
        if "(" in dir_name:
            judge_model, checklist_part = dir_name.split("(", 1)
            judge_model = judge_model.strip()
            checklist_type = checklist_part.split(")")[0].strip()
        else:
            judge_model = dir_name
            checklist_type = "default"
        
        # Process each JSON file in the directory
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                with open(os.path.join(dir_path, filename)) as f:
                    data = json.load(f)
                    for entry in data:
                        key = (entry["model_test"], entry["session_id"])
                        if key in gpt4o_scores:
                            results[judge_model][checklist_type]["total"] += 1
                            if entry["score"] == gpt4o_scores[key]:
                                results[judge_model][checklist_type]["agreed"] += 1
    return results

# Format results into table
def format_results(results):
    # Collect all checklist types
    checklist_types = set()
    for judge in results.values():
        checklist_types.update(judge.keys())
    
    # Sort checklist types with priority to main categories
    priority_order = ["GPT4o Checklist", "self-Checklist Generation", "CheckGen-v0-2"]
    sorted_types = sorted(checklist_types, 
                         key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))
    
    # Print header
    print("agreement w/ gpt-4o")
    header = ["judge_model"] + sorted_types
    print(" | ".join(header))
    
    # Print rows
    for judge in sorted(results.keys()):
        row = [judge]
        for ctype in sorted_types:
            if ctype in results[judge]:
                total = results[judge][ctype]["total"]
                agreed = results[judge][ctype]["agreed"]
                pct = f"{agreed/total*100:.0f}%" if total > 0 else "N/A"
                row.append(pct)
            else:
                row.append("N/A")
        print(" | ".join(row))

if __name__ == "__main__":
    gpt4o_scores = load_gpt4o_scores()
    agreement_results = calculate_agreement(gpt4o_scores)
    format_results(agreement_results)