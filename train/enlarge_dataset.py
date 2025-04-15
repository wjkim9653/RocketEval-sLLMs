from datasets import load_dataset, load_from_disk
from pprint import pprint
import json
from tqdm import tqdm
import os
import random
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


"""
# STEP 1
# Download LMSYS-Chat-1M dataset and process it (only ENG, drop openai_moderation column)
ds_original = load_dataset('lmsys/lmsys-chat-1m')
ds_english_only = ds_original['train'].filter(lambda x: x['language']=='English')
ds_english_only = ds_english_only.remove_columns(['openai_moderation'])

print(ds_english_only)
pprint(ds_english_only[0])
ds_english_only.save_to_disk("../data/lmsys_chat_1m_en")
"""



"""
# STEP 2
# Process!
def generate_prompt_samples(example):  # takes each row from dataset as parameter
    conversation = example["conversation"]  # contains multiple dict each w/ 'role' and 'content' keys
    samples = []
    turns = []  # contains tuples each with a (user dict + assistant dict) conversation pair

    # group conversation into user+assistant pairs
    for i in range(0, len(conversation)-1, 2):
        if conversation[i]['role'] == 'user' and conversation[i+1]['role'] == 'assistant':  # if correct pair
            turns.append((conversation[i], conversation[i+1]))
        else:
            # skip this example
            return []
    # turns will have all conversation pairs -> if 6 turn example was given, turns would be: [(pair1),(pair2),(pair3),(pair4),(pair5),(pair6)]
    
    for i in range(len(turns)):  # assuming i is the latest state of conversation
        history  = turns[:i]  # up until i-th conversation
        query_turn = turns[i][0]  # {'role':'user', 'content':'user asking something'}
        response_turn = turns[i][1]  # {'role':'assistant', 'content':'assistant answering it'}
        
        history_string = ""  # all history converstaion pairs (each a tuple of dicts) -> as string
        if history:
            history_string = "".join(
                f"<|start_header_id|>{turn[0]['role']}<|end_header_id|>\n\n{turn[0]['content']}<|eot_id|>"
                f"<|start_header_id|>{turn[1]['role']}<|end_header_id|>\n\n{turn[1]['content']}<|eot_id|>"
                for turn in history  # for each turns
            )
            history_strint = f'<|begin_of_history|>{history_string}<|end_of_history|>'
        
        query = f"<|begin_of_query|>{query_turn['content']}<|end_of_query|>"  # final turn's user query
        response = f"<|begin_of_reference_response|>{response_turn['content']}<|end_of_reference_response|>"  # final turn's assistant response

        full_text = (
            "<|begin_of_text|><|start_header_id|>user<end_header_id|>\n\n"
            "# Instruction\n"
            "You are an helpful assistant who identifies and summarizes key factors in large language models (LLMs) evaluation to help humans evaluate LLMs efficiently.\n\n"
            "Feed any query into different LLMs, I will get various responses. I need to know in quick whether these responses follows the instructions and answers the question in the user query better.\n\n"
            "I'll provide you with a user query. Your task is to identify those key factors that will affect my judgment and summarize them into a list to improve the efficiency of my evaluation.\n\n"
            "# Conversation between User and AI\n"
            f"{history_string}\n\n"
            "## Current User Query\n"
            f"{query}\n\n"
            "## Reference Response\n"
            f"{response}\n\n"
            "# Task\n"
            "Given the above information, I need you to create a binary question list, so that I can perform an efficient and accurate evaluation through answering several questions.\n\n"
            "Your question should be concise and include any necessary key content and information (such as keywords, formats, correct counts and values) in the user query or expected to be shown in responses. Your questions should not only consider evaluating the reference response, but all possible responses. Avoid creating duplicate, cumbersome or vague questions. For example, you should ask \"Is this response contain the correct answer ...\" instead of \"Is this response's answer correct?\". Ask fewer questions by aggregating questions with repeated contexts into one question.\n\n"
            "## Output Format\n"
            "Please provide your outputs in the following markdown format by filling in the placeholders in {}:\n"
            "```\n"
            "1. {question1}\n"
            "2. {question2}\n"
            "...\n"
            "```"
            "<|eot_id|><|start_header_id|>assistant<end_header_id|>"
            "#####INVOKE CHAT GPT API TO CREATE GOLD RESPONSE AND PUT IT HERE#####"
            "<|eot_id|><|end_of_text|>"
        )

        samples.append({'text': full_text})
    return samples

# Load cleaned dataset from disk(w/ ENG only, dropped openai_modearion column)
dataset = load_from_disk('data/lmsys_chat_1m_en')

all_samples = []
for example in tqdm(dataset, desc="processing..."):
    expanded = generate_prompt_samples(example)
    all_samples.extend(expanded)
random.shuffle(all_samples)
print(f"Generated {len(all_samples)} prompt-response pairs")


output_dir = "data/lmsys_chat_1m_en_expanded"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "lmsys_chat_1m_en_expand_no_gold.jsonl"), "w") as f:
    for sample in tqdm(all_samples, desc="saving..."):
        json.dump(sample, f)
        f.write("\n")

with open(os.path.join(output_dir, "lmsys_chat_1m_en_expand_no_gold_final.jsonl"), "w") as f:
    for sample in tqdm(all_samples[:10000], desc="saving..."):
        json.dump(sample, f)
        f.write("\n")

with open(os.path.join(output_dir, "lmsys_chat_1m_en_expand_no_gold_snippet.jsonl"), "w") as f:
    for sample in tqdm(all_samples[:30], desc="saving..."):
        json.dump(sample, f)
        f.write("\n")

print("done")
"""




"""
# STEP 3
# CREATE GOLD RESPONSES W/ GPT
enlarged_file_no_gold_dir = "data/lmsys_chat_1m_en_expanded/lmsys_chat_1m_en_expand_no_gold_final.jsonl"
enlarged_file_gold_dir = "data/lmsys_chat_1m_en_expanded/lmsys_chat_1m_en_expand_gold_final.jsonl"

front_snip = '<|begin_of_text|><|start_header_id|>user<end_header_id|>\n\n'
end_snip = '<|eot_id|><|start_header_id|>assistant<end_header_id|>#####INVOKE CHAT GPT API TO CREATE GOLD RESPONSE AND PUT IT HERE#####<|eot_id|><|end_of_text|>'

client = OpenAI()
def call_gpt(api_req_content:str):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role":"user", "content":api_req_content}]
        )
        reply = response.choices[0].message.content
        return api_req_content, reply
    except Exception as e:
        return api_req_content, f"[ERROR: {str(e)}]"

# 1. Read all prompts
api_req_contents = []
with open(enlarged_file_no_gold_dir, "r") as f_in, open(enlarged_file_gold_dir, "w") as f_out:
    for line in tqdm(f_in, desc="processing..."):
        data = json.loads(line)  # dict, {'text': '...'}
        text = data['text']  # string
        api_req_content = text[len(front_snip):-len(end_snip)]  # snippet of text for gpt api call
        api_req_contents.append(api_req_content)

# 2. Send api requests in parallel
api_req_results = []
with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(call_gpt, content) for content in api_req_contents]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Sending API Requests..."):
        api_req_content, reply = future.result()
        api_req_results.append((api_req_content, reply))

# 3. Save output to file
with open(enlarged_file_gold_dir, "w") as f_out:
    for api_req_content, reply in api_req_results:
        text_with_gold = (
            front_snip + api_req_content +
            '<|eot_id|><|start_header_id|>assistant<end_header_id|>' +
            reply + '<|eot_id|><|end_of_text|>'
        )
        json.dump({"text": text_with_gold}, f_out)
        f_out.write('\n')
print("done")
"""