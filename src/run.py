
import sys
import os

rocketeval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, rocketeval_dir)

import time
import openai
import logging
import textwrap
import argparse

from rich.logging import RichHandler
from rich.console import Console
from rich.markdown import Markdown

from rocketeval.data.data_loader import load_target_models
from rocketeval.task import checklist_task, judgment_task, ranking_task, score_task


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RocketEval Task Runner")
        
    # Data
    parser.add_argument("--data_dir", default="data/", help="Data directory")
    parser.add_argument("--config_dir", default="config/", help="Config directory")

    # Model
    parser.add_argument("--dataset", default="mt-bench", help="Dataset name")
    parser.add_argument("--generator", default="gpt-4o", help="Generator model")
    parser.add_argument("--judge", default="gpt-4o", help="Judge model")
    parser.add_argument("--labeler", default="gpt-4o", help="Labeler judge that provides labels")
    parser.add_argument("--train_test", action="store_true", help="Use specific train-test split")
    parser.add_argument("--gen_checklist", action="store_true", help="Whether to generate checklist")

    # Running Mode
    parser.add_argument("--mode", choices=["api", "offline"], help="Running mode, set to 'api' to use OpenAI API, set to 'offline' to use local models through vLLM")
    parser.add_argument("--instant_api", action="store_true", help="Run using instant API")
    parser.add_argument("--api_parallel_size", default=1, help="Number of parallel API calls, adjust based on your API rate limit.")
    parser.add_argument("--offline_config", default="config/offline/default.yaml", help="Path to offline vLLM engine config file")

    # Others
    parser.add_argument("--resume_from_task_id", default=None, help="To resume from a specific task with a given task ID")
    parser.add_argument("--keep_batch_files", action="store_true", help="Keep batch processing files after task finished")
    parser.add_argument("--gpu_ids", default="0", help="GPU IDs, split by comma")

    args = parser.parse_args()
    kwargs = vars(args)

    task_id = f"{args.dataset}_{int(time.time())}" \
        if args.resume_from_task_id is None \
        else args.resume_from_task_id

    if args.mode == "api":
        client = openai.OpenAI()
    else:
        client = None
        if not os.path.exists(os.path.join(args.data_dir, "batch")):
            os.makedirs(os.path.join(args.data_dir, "batch"))

    task_id = f"{args.dataset}_{int(time.time())}" \
        if args.resume_from_task_id is None \
        else args.resume_from_task_id
    
    train_model_names = load_target_models(
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        dataset_name=args.dataset,
        split="train" if args.train_test else "full"
    )

    test_model_names = load_target_models(
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        dataset_name=args.dataset,
        split="test" if args.train_test else "full"
    )

    start_message = textwrap.dedent(f"""[underline bold red on white blink]RocketEval[/]
    [bold yellow on red blink] Task Information[/]
    - Dataset: "{args.dataset}"
    - Judge: "{args.judge}"
    - Labeler: "{args.labeler}"
    - Task ID: "{task_id}"
    """)

    logger.info(start_message, extra={"markup": True})


    logger.info(f"[bold yellow on red blink]RocketEval Completed[/]", extra={"markup": True})

    if args.gen_checklist:
        # I - Checklist Creation
        logger.info(
            "[bold yellow on red blink]I. Checklist Creation[/]", extra={"markup": True}
        )

        checklist_task(
            client=client,
            task_id=task_id,
            **kwargs
        )

        logger.info(
            f"[yellow]Checklist Creation completed.[/]\n\n",
            extra={"markup": True}
        )
    else:
        logger.info(
            f"[bold yellow on red blink]Checklist Creation skipped.[/]",
            extra={"markup": True}
        )

    # II - Judgment Creation
    logger.info(
        "[bold yellow on red blink]II. Judgment Creation[/]", extra={"markup": True}
    )

    judgment_task(
        model_names=train_model_names + test_model_names,
        client=client,
        task_id=task_id,
        **kwargs
    )

    logger.info(
        f"[yellow]Judgment Creation completed.[/]\n\n",
        extra={"markup": True}
    )


    # III - Score Creation
    logger.info(
        f"[bold yellow on red blink]III. Score Creation[/]",
        extra={"markup": True}
    )

    score_task(
        train_model_names=train_model_names,
        test_model_names=test_model_names,
        task_id=task_id,
        **kwargs
    )

    logger.info(
        f"[yellow]Score Creation completed.[/]\n\n",
        extra={"markup": True}
    )


    # IV - Ranking
    logger.info(
        f"[bold yellow on red blink]IV. Ranking[/]",
        extra={"markup": True}
    )

    ranking = ranking_task(
        model_names=test_model_names,
        **kwargs
    )

    Console().print(Markdown(ranking.to_markdown()), justify="center")

    logger.info(
        f"[yellow]Ranking completed.[/]\n\n",
        extra={"markup": True}
    )


    # Finish
    logger.info(
        f"[bold yellow on red blink]RocketEval Completed[/]",
        extra={"markup": True}
    )