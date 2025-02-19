import os
import json
import openai
import logging
import time
import pandas as pd
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..tools.openai import run_batch_instance, batch_create

logger = logging.getLogger("rich")

def run_batch_openai(
    input_file: str,
    output_file: str | None = None,
    model: str | None = "google/Gemma-2-2B-it",
    client: openai.OpenAI | None = None,
    max_running_task: int = 3,
    **kwargs
) -> None:
    """
    Run batch inference on OpenAI API batch mode.
    Args:
        input_file: Input file path
        output_file: Output file path
        model: Model name
        client: OpenAI client
        max_running_task: Maximum number of concurrent running tasks
        **kwargs: Additional arguments
    """
    chunksize = kwargs.get("chunksize", 4096)

    if not output_file:
        output_file = input_file.rsplit(".", 1)[0] + "_results.jsonl"
    
    # Read all chunks first
    chunks = []
    for ind, input_chunk in enumerate(pd.read_json(input_file, lines=True, chunksize=chunksize)):
        input_chunk_file = input_file.replace(".jsonl", f"_chunk-{ind}.jsonl")
        output_chunk_file = output_file.replace(".jsonl", f"_chunk-{ind}.jsonl")
        input_chunk.to_json(input_chunk_file, orient='records', lines=True)
        chunks.append((ind, input_chunk_file, output_chunk_file))

    task_ids = {}
    pending_chunks = chunks.copy()
    
    def submit_task(chunk_info):
        ind, input_chunk_file, output_chunk_file = chunk_info
        task_id = batch_create(client, input_chunk_file)
        os.remove(input_chunk_file)
        task_ids[task_id] = {
            "status": False,
            "output_file": output_chunk_file
        }
        print(f"Task {task_id} created")
        return task_id

    # Submit initial batch of tasks up to max_running_task
    initial_tasks = pending_chunks[:max_running_task]
    pending_chunks = pending_chunks[max_running_task:]
    for chunk_info in initial_tasks:
        submit_task(chunk_info)
    
    running_time = time.time()

    while True:
        completed_tasks = []
        for task_id in task_ids:
            if task_ids[task_id]["status"]:
                continue
            
            task = client.batches.retrieve(task_id)
            if task.status in ["validating", "in_progress", "finalizing"]:
                pass
            elif task.status == "completed":
                file_response = client.files.content(task.output_file_id)
                with open(task_ids[task_id]["output_file"], "wb") as f:
                    f.write(file_response.content)
                task_ids[task_id]["status"] = True
                completed_tasks.append(task_id)
            else:
                raise Exception(f"Task {task_id} failed with status {task.status}")
        
        # Submit new tasks for each completed task if there are pending chunks
        for _ in completed_tasks:
            if pending_chunks:
                chunk_info = pending_chunks.pop(0)
                submit_task(chunk_info)
        
        if all([task_ids[task_id]["status"] for task_id in task_ids]) and not pending_chunks:
            break

        num_finished = sum([task_ids[task_id]["status"] for task_id in task_ids])
        num_running = len(task_ids) - num_finished
        print(f"{num_finished} tasks finished, {num_running} tasks running, {len(pending_chunks)} tasks pending. Time taken: {round((time.time() - running_time)/60, 1)} minutes")
        sleep(kwargs.get("update_interval", 60))
    
    print("All tasks finished, concatenating results...")
    open(output_file, "w").close()  # Clear the file
    for task_id in task_ids:
        pd.read_json(task_ids[task_id]["output_file"], lines=True).to_json(output_file, orient='records', lines=True, mode='a')
        os.remove(task_ids[task_id]["output_file"])
    print(f"""Results concatenated to "{output_file}" """)

  
def run_batch_instant_api(
    input_file: str,
    output_file: str | None = None,
    model: str | None = "google/Gemma-2-2B-it",
    client: openai.OpenAI | None = None,
    **kwargs
) -> None:
    """
    Run batch inference on OpenAI API instant mode.
    """

    api_parallel_size = int(kwargs.get("api_parallel_size", 1))
    if not output_file:
        output_file = input_file.rsplit(".", 1)[0] + "_results.jsonl"

    output = open(output_file, "a")

    def process_line(line):
        row = json.loads(line)
        result = run_batch_instance(row, client)
        return json.dumps(result) + "\n"

    with open(input_file, "r") as f:
        lines = f.readlines()

    if api_parallel_size > 1:
        with ThreadPoolExecutor(max_workers=api_parallel_size) as executor:
            futures = [executor.submit(process_line, line) for line in lines]
            for future in as_completed(futures):
                output.write(future.result())
    else:
        for line in lines:
            output.write(process_line(line))
    
    output.close()