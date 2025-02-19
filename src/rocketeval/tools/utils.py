import re
import numpy as np
from typing import List

question_pattern = re.compile(r'(?<=[0-9]\. ).+\?')

def extract_content(response: dict) -> str:
    """
    Extract the content from the given response.
    """
    if "body" in response:
        return response["body"]["choices"][0]["message"]["content"]
    elif "outputs" in response:
        return response["outputs"][0]["text"]
    else:
        return str(response)

def match_questions(response: str) -> List[str]:
    """
    Match checklist questions from the given responses.
    """
    response = response.split("##")[-1]
    questions = question_pattern.findall(response) if question_pattern.findall(response) else [""]
    return questions

def extract_info(row: dict) -> dict:
    """
    Extract the necessary information from the given row (WildBench format).
    """
    history = chat_history(row["conversation_input"])
    instruction = row["conversation_input"][-1]["content"]
    ref_response = row["references"]["gpt-4"]
    return {
        "history": history,
        "instruction": instruction,
        "reference_response": ref_response
    }

def chat_history(conversation_input) -> dict:
    """
    Convert the conversation input to chat history as the context.
    """
    history = ""
    if len(conversation_input) > 0: 
        for x in conversation_input[:-1]:
            if x["role"] == "user":
                history += "USER: " + x["content"] + "\n\n"
            elif x["role"] == "assistant":
                history += "ASSISTANT: " + x["content"] + "\n\n"
    return history

def robust_logprobs(logprobs: list) -> dict:
    """
    Extract the logprobs of target tokens from the given logprobs list.
    """
    prob = {
        "Yes": float("-inf"),
        "No": float("-inf")
    }
    for lobprob in logprobs:
        if "token" in lobprob:
            word = lobprob['token'].strip().capitalize()
        else:
            word = lobprob['decoded_token'].strip().capitalize()
        if word in prob:
            prob[word] = max(prob[word], lobprob["logprob"])
    return prob

def get_confidence(prob: dict, default: int | float = np.nan) -> str:
    """
    Derive the conditional normalized probability from the judgment response.
    """
    if prob["Yes"] == prob["No"] == float("-inf"):
        return default
    else:
        confidence = np.exp(prob["Yes"]) / (np.exp(prob["Yes"]) + np.exp(prob["No"]))
        return confidence

def get_judgment(score: float) -> str:
    """
    Derive the judgment from the score.
    """
    if np.isnan(score) or abs(score - 0.5) < 1e-3:
        return "Unsure"
    else:
        judgment = "Yes" if score > 0.5 else "No"
        return f"{judgment} ({score * 100:.1f})%"

def get_all_judgments(checklist: dict) -> list:
    """
    Derive all judgment from the checklist.
    """
    judgment_list = ["Unsure"] * (max(checklist.keys()) + 1)
    for key in checklist.keys():
        judgment_list[key] = get_judgment(checklist[key])
    return judgment_list

def get_confidence_score(item: dict) -> float:
    """
    Derive the confidence score from the judgment response.
    """
    
    try:
        if "body" in item["response"]:
            logprob = item["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        elif "outputs" in item["response"]:
            logprob = list(item["response"]["outputs"][0]["logprobs"][0].values())
    except Exception as e:
        logprob = []
        
    choice = robust_logprobs(logprob)
    choice = get_confidence(choice)
    return choice