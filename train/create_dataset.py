import json
import argparse
import os
import random
from glob import glob

def combine_two_files(file1, file2, output_file):
    count = 0
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as out:
        for row1_line, row2_line in zip(f1, f2):
            row1 = json.loads(row1_line)
            row2 = json.loads(row2_line)

            result1 = row1["response"]["prompt"]
            result2 = row2["checklist"]

            checklist_str = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(result2))
            formatted = f"{result1}\n```\n{checklist_str}\n```<|eot_id|>"

            out.write(json.dumps({"text": formatted}) + '\n')
            count += 1

    print(f"Generated {count} JSON lines in '{output_file}'.")

def merge_and_shuffle_and_split(jsonl_sources, output_file):
    all_files = []

    # Collect all .jsonl files from directories or direct file inputs
    for source in jsonl_sources:
        if os.path.isdir(source):
            all_files.extend(glob(os.path.join(source, "*.jsonl")))
        elif os.path.isfile(source) and source.endswith(".jsonl"):
            all_files.append(source)

    print(f"Found {len(all_files)} JSONL files to merge and shuffle.")

    all_lines = []
    for filepath in all_files:
        with open(filepath, 'r') as f:
            for line in f:
                all_lines.append(json.loads(line.strip()))

    print(f"Loaded {len(all_lines)} total JSON lines. Shuffling...")

    random.shuffle(all_lines)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as out:
        for item in all_lines:
            out.write(json.dumps(item) + '\n')

    print(f"Merged & shuffled data saved to '{output_file}'.")

    # split and save trainset(90%)/validationset(10%)
    split_idx = int(len(all_lines) * 0.9)
    train_set = all_lines[:split_idx]
    val_set = all_lines[split_idx:]

    train_path = output_file.replace(".jsonl", "_train.jsonl")
    val_path = output_file.replace(".jsonl", "_val.jsonl")

    with open(train_path, 'w') as f:
        for item in train_set:
            f.write(json.dumps(item) + '\n')
    print(f"Train set saved to '{train_path}' with {len(train_set)} entries.")

    with open(val_path, 'w') as f:
        for item in val_set:
            f.write(json.dumps(item) + '\n')
    print(f"Validation set saved to '{val_path}' with {len(val_set)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process or merge/shuffle JSONL datasets.")
    parser.add_argument('--file1', type=str, help='Path to the first JSONL file (with prompt).')
    parser.add_argument('--file2', type=str, help='Path to the second JSONL file (with checklist).')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output JSONL.')

    parser.add_argument('--merge_and_shuffle_and_split', nargs='+', help='List of JSONL files or directories to merge and shuffle.')

    args = parser.parse_args()

    if args.merge_and_shuffle_and_split:
        merge_and_shuffle_and_split(args.merge_and_shuffle_and_split, args.output_file)
    elif args.file1 and args.file2:
        combine_two_files(args.file1, args.file2, args.output_file)
    else:
        print("❌ Error: Either provide --file1 and --file2 or use --merge_and_shuffle.")


'''
$~ python train/create_dataset.py \
    --file1 "data/batch/alpacaeval_(Meta-Llama-3-8B-Instruct)_checklist.batch_results.jsonl" \
    --file2 data/alpacaeval/checklist/checklist_baseline_gpt4o.jsonl \
    --output_file data/train_data/alpacaeval_trainset.jsonl

$~ python train/create_dataset.py \
    --file1 "data/batch/arena-hard_(Meta-Llama-3-8B-Instruct)_checklist.batch_results.jsonl" \
    --file2 data/arena-hard/checklist/checklist_baseline_gpt4o.jsonl \
    --output_file data/train_data/arena-hard_trainset.jsonl

$~ python train/create_dataset.py \
    --file1 "data/batch/mt-bench_(Meta-Llama-3-8B-Instruct)_checklist.batch_results.jsonl" \
    --file2 data/mt-bench/checklist/checklist_baseline_gpt4o.jsonl \
    --output_file data/train_data/mt-bench_trainset.jsonl

$~ python train/create_dataset.py \
    --file1 "data/batch/wildbench_(Meta-Llama-3-8B-Instruct)_checklist.batch_results.jsonl" \
    --file2 data/wildbench/checklist/checklist_baseline_gpt4o.jsonl \
    --output_file data/train_data/wildbench_trainset.jsonl

$~ python train/create_dataset.py \
    --merge_and_shuffle_and_split data/train_data/alpacaeval_trainset.jsonl data/train_data/arena-hard_trainset.jsonl data/train_data/mt-bench_trainset.jsonl data/train_data/wildbench_trainset.jsonl \
    --output_file data/train_data/trainset_merged.jsonl
'''