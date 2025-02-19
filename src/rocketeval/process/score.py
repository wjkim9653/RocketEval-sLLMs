import os
import logging
import numpy as np
from tqdm import tqdm
from typing import List

from collections import Counter

from scipy.special import kl_div
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor

from ..data.data_loader import load_results_data, load_judgment_data, load_score_data

logger = logging.getLogger("rich")

class Mean(BaseEstimator):
    """
    Mean score predictor.
    """
    def __init__(self, scale=9, bias=1):
        super().__init__()
        self.scale = scale
        self.bias = bias

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.nanmean(np.array(X), axis=1) * self.scale + self.bias

def score_hist(scores: list):
    """
    Compute the histogram of the given score distribution.
    """
    counts = Counter(scores)
    hist = np.array(list(counts.values()))
    return hist / hist.sum()

def kl_uniform_weight(X: np.ndarray, num_classes: int = 10):
    """
    Compute the KL-divergence based weight for the given score distribution.
    This weight determines how much to rely on the trained model vs the mean baseline.
    A higher weight means the distribution is more uniform (diverse scores).
    """
    # Create uniform distribution for comparison
    uniform_dist = np.ones(num_classes) / num_classes
    # Calculate maximum possible KL divergence (when all scores are in one bin)
    max_kl = kl_div([1.0] + [0.0] * (num_classes - 1), uniform_dist).sum()
    # Get actual score distribution
    hist = score_hist(X)
    padding = num_classes - hist.shape[0]
    assert padding >= 0
    # Return normalized weight (1 - actual_kl/max_kl) so more uniform distributions get higher weights
    return (max_kl - kl_div(np.pad(hist, (0, padding), 'constant'), uniform_dist).sum()) / max_kl

def fit_score(X: np.ndarray, y: np.ndarray):
    """
    Learn the supervised score predictor.
    """
    model = ExtraTreesRegressor(
        n_estimators=10,
        max_depth=2,
        random_state=42)
    model.fit(X, y)
    return model

def predict_score(model: BaseEstimator,  X: np.ndarray, weight: float = 0):
    """
    Predict the final score.
    """
    return model.predict(X) * weight + Mean().predict(X) * (1 - weight)

def create_score(
    dataset_name: str = "wildbench",
    judge: str = "gpt-4o",
    labeler: str = "gpt-4o",
    train_model_names: List[str] | None = [],
    test_model_names: List[str] = ['Meta-Llama-3-8B-Instruct'],
    data_dir: str = "data/",
    task_id: str = "",
    add_output: bool = False
) -> None:
    """
    Create score file for the given dataset.
    This function implements a hybrid scoring approach:
    1. For test samples, it loads judgment data (model probabilities)
    2. If training models are provided:
       - Trains a model on their scores
       - Uses KL divergence to determine weight between trained model and mean baseline
    3. Otherwise defaults to using just the mean baseline
    4. Saves per-model scores to separate JSON files
    """
    # Load test judgments and convert to array format
    test_judgment = load_judgment_data(data_dir=data_dir, dataset_name=dataset_name, judge=judge, model_names=test_model_names).set_index(["session_id"])
    test_sample = test_judgment.groupby("session_id").agg(list).map(np.array)

    if train_model_names and len(train_model_names) > 0:
        # Load training data if provided - includes both judgments and ground truth scores
        train_judgment = load_judgment_data(data_dir=data_dir, dataset_name=dataset_name, judge=judge, model_names=train_model_names).set_index(["session_id", "model_test"])
        train_label = load_score_data(data_dir=data_dir, dataset_name=dataset_name, model_names=train_model_names, judge=labeler).set_index(["session_id", "model_test"])
        
        # Prepare training samples and fit scorer for each session
        train_sample = train_judgment.join(train_label, how="inner")
        train_sample = train_sample.loc[:, ["norm_probability", "score"]].groupby("session_id").agg(list).map(np.array)
        # Train individual scorer for each session using probabilities and scores
        train_sample["scorer"] = train_sample.apply(lambda x: fit_score(x["norm_probability"], x["score"]), axis=1)
        # Calculate weights based on score distribution uniformity
        train_sample["weight"] = train_sample["score"].apply(lambda x: kl_uniform_weight(x))
        test_sample = test_sample.join(train_sample.loc[:, ["scorer", "weight", "score"]], how="inner")
    else:
        # If no training data, use simple mean baseline
        test_sample["scorer"] = Mean()
        test_sample["weight"] = 0

    # Generate final scores using weighted combination of trained model and mean baseline
    test_sample["score"] = test_sample.apply(lambda x: predict_score(x["scorer"], x["norm_probability"], x["weight"]), axis=1)
    test_sample = test_sample.loc[:, ["model_test", "score"]].explode(["model_test", "score"]).reset_index().set_index(["session_id", "model_test"])

    if add_output:
        results = load_results_data(data_dir=data_dir, dataset_name=dataset_name, model_names=test_model_names).set_index(["session_id", "model_test"]).rename(columns={"output": "model_output"})
        results["model_output"] = results["model_output"].apply(lambda x: x[0])
        test_sample = test_sample.join(results.loc[:, ["model_output"]], how="inner").reset_index()
    else:
        test_sample["model_output"] = None
        test_sample = test_sample.reset_index()
    test_sample["judge"] = judge
    test_sample["task_id"] = task_id

    output_dir = os.path.join(data_dir, dataset_name, "score", judge)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for model in tqdm(test_model_names):
        model_test_sample = test_sample[test_sample["model_test"] == model]
        output_file = os.path.join(output_dir, f"{model}.json")
        model_test_sample.to_json(output_file, orient='records', indent=2)
        logger.info(f"""Score of "{model}" output to "{output_file}" """)
