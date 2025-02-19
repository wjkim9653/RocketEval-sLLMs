import os
import logging
import pandas as pd

from ..data.data_loader import load_bench_data
from ..tools.openai import row_create, prompt_to_message
from ..tools.utils import chat_history, extract_content, match_questions

logger = logging.getLogger("rich")

def create_checklist(
    dataset_name: str,
    output_file: str = None,
    model: str = 'gpt-4o',
    data_dir: str = "data/",
    config_dir: str = "config/",
    task_id: str = ""
) -> str:
    """
    Generate checklist creating batch file for the given dataset.
    """
    # Load the template that defines the structure for checklist creation
    template = open(os.path.join(config_dir, "template", "create.md"), 'r').read()
    
    # Load benchmark data and prepare conversation history
    data = load_bench_data(dataset_name=dataset_name)
    # Convert conversation inputs into a structured chat history format
    data["history"] = data["conversation_input"].apply(chat_history)
    # Extract the last user message from the conversation
    data["user_query"] = data["conversation_input"].apply(lambda x: x[-1]["content"])
    # Get GPT-4's response as reference
    data["reference_response"] = data["references"].apply(lambda x: x["gpt-4"])
    
    # Format the template with data from each row to create prompts
    data['prompt'] = data.apply(lambda x: template.format(**x), axis=1)
    # Convert prompts into message format suitable for API calls
    data['messages'] = data['prompt'].apply(prompt_to_message)
    
    # Create batch data by applying model parameters to each row
    batch_data = data.apply(
        lambda x: row_create(
            model=model,
            custom_id=x["session_id"],
            messages=x["messages"],
            temperature=0.7,  # Controls randomness in model output
            max_tokens=1024,  # Maximum length of generated response
            top_p=0.95       # Nucleus sampling parameter for response diversity
        ), axis=1)
    
    # Set default output path if not specified
    if not output_file:
        output_file = os.path.join(data_dir, "batch", f"{task_id}_checklist.batch_submission.jsonl")
    # Save batch data as JSONL file
    batch_data.to_json(output_file, orient='records', lines=True)
    logger.info(f"""Batch file output to "{output_file}" """)
    return output_file


def parse_checklist(
    dataset_name: str,
    input_file: str,
    data_dir: str = "data/",
) -> None:
    """
    Parse the checklist creating batch results and save to the target directory.
    """
    # Load batch results from JSONL file
    data = pd.read_json(input_file, lines=True, orient='records')
    # Extract clean content from model responses
    data['content'] = data['response'].apply(extract_content)
    # Parse the content to identify and structure checklist questions
    data['checklist'] = data['content'].apply(match_questions)
    data['session_id'] = data['custom_id']
    
    # Ensure output directory exists
    output_dir = os.path.join(data_dir, dataset_name, "checklist")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save processed checklist data, keeping only relevant columns
    output_file = os.path.join(output_dir, "checklist.jsonl")
    data.loc[:, ['session_id', 'checklist']].to_json(output_file, orient='records', lines=True)
    logger.info(f"""Checklist output to "{output_file}" """)
    return
