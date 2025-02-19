import re
import os
import logging
import pandas as pd
from tqdm import tqdm
from typing import List

from ..data.data_loader import load_bench_data, load_results_data
from ..tools.openai import row_create, prompt_to_message
from ..tools.utils import chat_history, get_confidence_score, get_judgment

# Regular expression to extract questions that follow a number and period (e.g., "1. What is...?")
question_pattern = re.compile(r'(?<=[0-9]\. ).+\?')
logger = logging.getLogger("rich")

def create_judgment(
    dataset_name: str,
    output_file: str = None,
    model_names: List[str] = ['Meta-Llama-3-8B-Instruct'],
    judge: str = 'gpt-4o',
    data_dir: str = "data/",
    config_dir: str = "config/",
    task_id: str = ""
) -> str:
    """
    Create judgment batch file for the given dataset.
    """
    # Load the grading template that will be used to format prompts for judgment
    template = open(os.path.join(config_dir, "template", "grading.md"), 'r').read()
    bench_data = load_bench_data(dataset_name=dataset_name, load_checklist=True)
    inst_cnt = len(bench_data)
    logger.info(f"""Loaded {inst_cnt} instances from '{dataset_name}'""")

    # Explode the checklist into separate rows, creating a row for each checklist item
    bench_data["checklist_id"] = bench_data["checklist"].apply(lambda x: list(range(len(x))))
    bench_data = bench_data.explode(["checklist", "checklist_id"]).rename(columns={"checklist": "question"}).set_index("session_id")
    
    # Process conversation data to extract chat history and relevant information
    bench_data["history"] = bench_data["conversation_input"].apply(chat_history)
    bench_data["user_query"] = bench_data["conversation_input"].apply(lambda x: x[-1]["content"])
    bench_data["reference_response"] = bench_data["references"].apply(lambda x: x["gpt-4"])
    question_cnt = len(bench_data)
    logger.info(f"""Generated {len(bench_data)} questions for judgment, average No. of questions per instance: {question_cnt/inst_cnt}""")

    # Load model outputs for comparison
    results_data = load_results_data(dataset_name=dataset_name, model_names=model_names)
    logger.info(f"""Loaded {len(results_data)} results from {len(model_names)} models""")

    results_data = results_data.set_index("session_id")
    model_batch_data = {}
    
    # Process each model's outputs and prepare them for judgment
    for model_test, model_data in tqdm(results_data.groupby("model_test")):
        # Extract the first output from model responses
        model_data["model_output"] = model_data["output"].apply(lambda x: x[0])
        
        # Join model outputs with benchmark data
        model_data = model_data.join(bench_data, how="inner", rsuffix="_bench")
        
        # Create unique identifiers for each judgment task
        model_data["custom_id"] = model_data.apply(lambda x: f"{x.name}||{x['model_test']}||{x['checklist_id']}", axis=1)
        model_data = model_data.set_index("custom_id")
        
        # Format prompts using the template and convert to message format
        model_data["prompt"] = model_data.apply(lambda x: template.format(**x), axis=1)
        model_data["messages"] = model_data["prompt"].apply(prompt_to_message)
        model_data.reset_index(inplace=True)
        
        # Create batch requests for the judge model
        batch_data = model_data.apply(
            lambda x: row_create(
                model=judge,
                custom_id=x["custom_id"],
                messages=x["messages"],
                temperature=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=5
            ), axis=1)
        model_batch_data[model_test] = batch_data

    # Save the batch submission file
    if not output_file:
        output_file = os.path.join(data_dir, "batch", f"{task_id}_judgment.batch_submission.jsonl")
    pd.concat(list(model_batch_data.values())).to_json(output_file, orient='records', lines=True)
    logger.info(f"""Batch file output to "{output_file}" """)
    return output_file

def parse_judgment(
    dataset_name: str,
    input_file: str,
    model_names: List[str] = ['Meta-Llama-3-8B-Instruct'],
    judge: str = 'gpt-4o',
    data_dir: str = "data/"
) -> None:
    """
    Parse the judgment batch results and save to the target directory.
    """
    # Load and parse the batch results
    results = pd.read_json(input_file, lines=True, orient='records')
    
    # Extract components from the custom_id (session_id, model_test, checklist_id)
    results["session_id"] = results["custom_id"].apply(lambda x: x.split("||")[0])
    results["model_test"] = results["custom_id"].apply(lambda x: x.split("||")[1])
    results["checklist_id"] = results["custom_id"].apply(lambda x: x.split("||")[2])
    results = results[results["checklist_id"].apply(str.isdigit)]

    # Calculate normalized probabilities and convert to judgment scores
    results["norm_probability"] = results.apply(get_confidence_score, axis=1)
    results["judgment"] = results["norm_probability"].apply(get_judgment)

    # Process and save results for each model
    for model in tqdm(model_names):
        # Filter results for current model and aggregate by session
        model_results = results[results["model_test"] == model]
        model_results = model_results.sort_values(["model_test", "session_id", "checklist_id"]).groupby(["model_test", "session_id"]).agg(list)
        model_results = model_results.loc[:, ["norm_probability", "judgment"]].reset_index()
        model_results["judge"] = judge

        # Create output directory and save results
        save_path = os.path.join(data_dir, dataset_name, "judgment", judge)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        output_file = os.path.join(save_path, f"{model}.jsonl")
        model_results.to_json(output_file, orient='records', lines=True)
    logger.info(f"""Judgment results output to "{save_path}" """)
