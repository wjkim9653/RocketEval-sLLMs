import random
import pandas as pd
from typing import List

from ..data.data_loader import load_score_data

def all_compare(
    model: List[str],
    score: List[float],
    tie_margin: float = 0.1,
) -> dict[str]:
    results = []
    for i in range(len(model)):
        for j in range(i + 1, len(model)):
            if random.random() > 0.5:
                model_a, model_b = model[i], model[j] 
                score_a, score_b = score[i], score[j]
            else:
                model_b, model_a = model[i], model[j]
                score_b, score_a = score[i], score[j]
                
            if score_a > score_b + tie_margin: winner = "model_a"
            elif score_b > score_a + tie_margin: winner = "model_b"
            else: winner = "tie"
            results.append({"model_a": model_a, "model_b": model_b, "winner": winner})
    return results

def single_compare(
    model: List[str],
    score: List[float],
    target_model: str,
    target_score: float,
    tie_margin: float = 0.1,
) -> dict[str]:
    """
    Compare list 
    """
    results = []
    for i in range(len(model)):
        if model[i] == target_model: continue
        score = score[i]
        if score > target_score + tie_margin: winner = "model_a"
        elif target_score > score + tie_margin: winner = "model_b"
        else: winner = "tie"
        results.append({"model_a": model[i], "model_b": target_model, "winner": winner})
    return results

def chatbot_arena_sample(**kwargs):
    sample = {
        "model_a": "",
        "model_b": "",
        "winner": "",
        "judge": "",
        "turn": 1,
        "anony": True,
        "language": "English",
        "tstamp": 0,
        "conv_metadata": {'sum_user_tokens': 8, 'sum_assistant_a_tokens': 256, 'sum_assistant_b_tokens': 256, 'context_a_tokens': 8, 'context_b_tokens': 8, 'turns': 1},
        "is_code": False,
        "is_refusal": False,
        "dedup_tag": {'high_freq': False, 'sampled': True},
        "category_tag": {
            'if_v0.1': {'if': True, 'score': 4},
            'math_v0.1': {'math': False},
            'criteria_v0.1': {'specificity': True, 'domain_knowledge': True, 'complexity': True, 'problem_solving': True, 'creativity': True, 'technical_accuracy': True, 'real_world': True}
        }
    }
    sample.update(**kwargs)
    sample.update(**{'tstamp': random.randint(0, 1000000)})
    return sample

def chatbot_arena_match(
    dataset_name: str,
    judge: str,
    model_names: List[str] = ['Meta-Llama-3-8B-Instruct'],
    data_dir: str = "data/",
) -> pd.DataFrame:
    """
    Export chatbot arena match data for the given dataset.
    """
    random.seed(0)

    data = load_score_data(
        data_dir=data_dir,
        dataset_name=dataset_name,
        judge=judge,
        model_names=model_names
    )

    data = data.groupby("session_id").agg({"model_test": list, "score": list})

    matches = data.apply(lambda x: all_compare(x["model_test"], x["score"]), axis=1)
    matches = pd.json_normalize(matches.explode())
    matches["judge"] = judge

    battles = pd.json_normalize(matches.apply(lambda x: chatbot_arena_sample(**x), axis=1), max_level=0)
    return battles
