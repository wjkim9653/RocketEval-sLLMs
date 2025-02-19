import os
import logging
import pandas as pd
from typing import List

from ..data.data_loader import load_score_data

logger = logging.getLogger("rich")

def create_ranking(
    dataset_name: str = "wildbench",
    judge: str = "gpt-4o",
    model_names: List[str] = ['Meta-Llama-3-8B-Instruct'],
    data_dir: str = "data/",
    task_id: str = "",
) -> pd.DataFrame:
    """
    Create ranking file for the given dataset.
    """
    # Load evaluation scores for specified models and judge
    results = load_score_data(dataset_name=dataset_name, judge=judge, model_names=model_names, data_dir=data_dir)
    
    # Calculate mean scores per model and sort in descending order
    # This creates a ranking based on average performance across all tasks
    results = results.groupby("model_test")["score"].mean().sort_values(ascending=False).to_frame(name=judge)
    
    # Ensure output directory exists
    output_dir = os.path.join(data_dir, "ranking")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Add rank numbers (1-based) and restructure DataFrame
    results['Rank'] = range(1, len(results) + 1)
    results = results.reset_index().set_index("Rank")
    
    # Save rankings to TSV file with task identifier
    results.to_csv(os.path.join(output_dir, f"{task_id}.tsv"), sep="\t")
    logger.info(f"""Ranking output to "{output_dir}/{task_id}.tsv" """)
    return results

def compare_ranking(
    dataset_name: str = "wildbench",
    judges: List[str] = ['gpt-4o'],
    model_names: List[str] = ['Meta-Llama-3-8B-Instruct'],
    data_dir: str = "data/",
) -> pd.DataFrame:
    """
    Compare different rankings from different judges.
    """
    # Initialize rankings DataFrame
    rankings = None
    
    # Iterate through each judge and combine their rankings
    for judge in judges:
        # Get scores and calculate mean performance per model for current judge
        ranking = load_score_data(dataset_name=dataset_name, judge=judge, model_names=model_names, data_dir=data_dir)
        ranking = ranking.groupby("model_test")["score"].mean().sort_values(ascending=False).to_frame(name=judge)
        
        # Merge with existing rankings using outer join to include all models
        # First iteration: rankings is None, so just use current ranking
        # Subsequent iterations: join with existing rankings
        rankings = rankings.join(ranking, how="outer") if rankings is not None else ranking
    return rankings