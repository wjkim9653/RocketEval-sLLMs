import os
import pandas as pd
from typing import List, Literal

def load_bench_data(
    data_dir: str = "data/",
    dataset_name: str = 'wildbench',
    load_checklist: bool = False
) -> pd.DataFrame:
    """
    Load the benchmark query data for the given dataset.
    """
    dataset = pd.read_json(os.path.join(data_dir, dataset_name, "query.jsonl"), lines=True)
    if load_checklist:
        checklist = pd.read_json(os.path.join(data_dir, dataset_name, "checklist", f"checklist.jsonl"), lines=True)
        dataset = dataset.drop(columns=['checklist']).join(checklist.set_index('session_id'), on='session_id', how='inner').fillna("")
    return dataset

def load_results_data(
    data_dir: str = "data/",
    dataset_name: str = 'wildbench',
    model_names: str | List[str] = 'Meta-Llama-3-8B-Instruct'
) -> pd.DataFrame:
    """
    Load the benchmark response data for the given dataset.
    """
    dataset = []
    for model_name in model_names:
        file_name = os.path.join(data_dir, dataset_name, "response", f"{model_name}.jsonl")
        if os.path.exists(file_name):
            model_data = pd.read_json(file_name, lines=True)
            model_data["model_test"] = model_name
            dataset.append(model_data)
        else:
            print(f"File {file_name} does not exist.")
    dataset = pd.concat(dataset)
    dataset["model_test"] = dataset["model_test"].apply(lambda x: x.split("/")[-1])     # Naming convention of WildBench
    return dataset

def load_judgment_data(
    data_dir: str = "data/",
    dataset_name: str = 'wildbench',
    judge: str = 'gpt-4o',
    model_names: str | List[str] = 'Meta-Llama-3-8B-Instruct',
    fillna: float = 0.5
) -> pd.DataFrame:
    """
    Load the benchmark judgment data for the given dataset.
    """
    dataset = []
    for model_name in model_names:
        file_name = os.path.join(data_dir, dataset_name, "judgment", judge, f"{model_name}.jsonl")
        if os.path.exists(file_name):
            dataset.append(pd.read_json(file_name, lines=True))
        else:
            print(f"File {file_name} does not exist.")
    dataset = pd.concat(dataset)
    dataset["norm_probability"] = dataset["norm_probability"].apply(lambda x: [fillna if e is None else e for e in x])
    return dataset

def load_score_data(
    data_dir: str = "data/",
    dataset_name: str = 'wildbench',
    judge: str | None = None,
    model_names: str | List[str] = 'Meta-Llama-3-8B-Instruct'
) -> pd.DataFrame:
    """
    Load the benchmark score data for the given dataset.
    """
    dataset = []
    for model_name in model_names:
        file_name = os.path.join(data_dir, dataset_name, "score", judge, f"{model_name}.json")
        if os.path.exists(file_name):
            dataset.append(pd.read_json(file_name))
        else:
            print(f"File {file_name} does not exist.")
    dataset = pd.concat(dataset)
    dataset["model_test"] = dataset["model_test"].apply(lambda x: x.split("/")[-1])     # Naming convention of WildBench
    dataset["score"] = dataset["score"].apply(float)
    return dataset

def load_target_models(
    data_dir: str = "data/",
    config_dir: str = "config/",
    dataset_name: str = 'wildbench',
    split: Literal["train", "test", "full"] = "test"
) -> List[str]:
    """
    Load the target running models for the given dataset.
    """
    if split == "full":
        models = []
        for file in os.listdir(os.path.join(data_dir, dataset_name, "response")):
            if file.endswith(".jsonl"):
                models.append(file.rsplit(".", 1)[0])
        return models
    else:
        model_file = os.path.join(config_dir, "rankings", f"{dataset_name}_{split}.json")
        try:
            models = pd.read_json(model_file)
            if "rating" not in models.columns:
                models["rating"] = 0
            return models.sort_values('rating', ascending=False)['name'].to_list()
        except:
            return []