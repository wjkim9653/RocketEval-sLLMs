import os
import openai
import pandas as pd
from typing import Callable

from .process.score import create_score
from .process.rank import create_ranking
from .process.judgment import create_judgment, parse_judgment
from .process.checklist import create_checklist, parse_checklist
from .tools.batch import run_batch_openai, run_batch_instant_api
from .tools.offline import run_batch_offline

def select_batch_runner(
        mode: str,
        client: openai.OpenAI | None = None,
        instant_api: bool = False
    ) -> Callable:
    if instant_api:
        if not client:
            raise ValueError("Running Instant API requires OpenAI client.")
        batch_runner = run_batch_instant_api
    else:
        batch_runner = run_batch_openai

    if mode == "offline":
        batch_runner = run_batch_offline
    return batch_runner

def checklist_task(
    dataset: str,
    generator: str = "gpt-4o",
    client: openai.OpenAI | None = None,
    data_dir: str = "data/",
    config_dir: str = "config/",
    task_id: str = "",
    keep_batch_files: bool = False,
    instant_api: bool = False,
    mode: str = "api",
    **kwargs
) -> None:
    """
    Run checklist task for the given dataset.
    """
    submission_file = create_checklist(
        dataset_name=dataset,
        model=generator,
        data_dir=data_dir,
        config_dir=config_dir,
        task_id=task_id
    )

    batch_runner = select_batch_runner(mode, client, instant_api)

    existing_vars = ["model", "client", "input_file", "output_file"]
    for var in existing_vars:
        kwargs.pop(var, None)

    batch_runner(
        input_file=submission_file,
        output_file=submission_file.replace("submission", "results"),
        model=generator,
        client=client,
        **kwargs
    )

    parse_checklist(
        dataset_name=dataset,
        input_file=submission_file.replace("submission", "results"),
        data_dir=data_dir
    )
    if not keep_batch_files:
        os.remove(submission_file)
        os.remove(submission_file.replace("submission", "results"))


def judgment_task(
    dataset: str,
    model_names: list[str],
    judge: str = "gpt-4o",
    client: openai.OpenAI | None = None,
    data_dir: str = "data/",
    task_id: str = "",
    keep_batch_files: bool = False,
    instant_api: bool = False,
    mode: str = "api",
    **kwargs
) -> None:
    """
    Run judgment task for the given dataset.
    """
    submission_file = create_judgment(
        dataset_name=dataset,
        judge=judge,
        model_names=model_names,
        data_dir=data_dir,
        task_id=task_id
    )

    batch_runner = select_batch_runner(mode, client, instant_api)

    existing_vars = ["model", "client", "input_file", "output_file"]
    for var in existing_vars:
        kwargs.pop(var, None)

    batch_runner(
        input_file=submission_file,
        output_file=submission_file.replace("submission", "results"),
        model=judge,
        client=client,
        **kwargs
    )

    parse_judgment(
        judge=judge,
        dataset_name=dataset,
        model_names=model_names,
        input_file=submission_file.replace("submission", "results")
    )
    if not keep_batch_files:
        os.remove(submission_file)
        os.remove(submission_file.replace("submission", "results"))


def score_task(
    dataset: str,
    train_model_names: list[str],
    test_model_names: list[str],
    judge: str = "google/Gemma-2-2B-it",
    labeler: str = "gpt-4o",
    data_dir: str = "data/",
    task_id: str = "",
    **kwargs
) -> None:
    """
    Run score task for the given dataset.
    """
    create_score(
        dataset_name=dataset,
        judge=judge,
        labeler=labeler,
        train_model_names=train_model_names,
        test_model_names=test_model_names,
        data_dir=data_dir,
        task_id=task_id,
        add_output=False
    )


def ranking_task(
    dataset: str,
    model_names: list[str],
    judge: str = "google/Gemma-2-2B-it",
    data_dir: str = "data/",
    task_id: str = "",
    **kwargs
) -> pd.DataFrame:
    """
    Run ranking task for the given dataset.
    """
    ranking = create_ranking(
        dataset_name=dataset,
        judge=judge,
        model_names=model_names,
        data_dir=data_dir,
        task_id=task_id
    )
    return ranking