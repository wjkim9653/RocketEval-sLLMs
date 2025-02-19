<h1></h1>

<h1 align="center">🚀 RocketEval 🚀</h1>

<h3 align="center">🚀 [ICLR '25] RocketEval: Efficient Automated LLM Evaluation via Grading Checklist</h3>

<p align="center">
<a href="https://openreview.net/forum?id=zJjzNj6QUe"><img src="https://img.shields.io/badge/OpenReview-zJjzNj6QUe-red"></a>
<!-- <a href="https://arxiv.org/abs/25XX.XXXXX><img src="https://img.shields.io/badge/arXiv-25XX.XXXXX-orange"></a> -->
<a href="https://github.com/Joinn99/RocketEval-ICLR/blob/master/LICENSE.md"><img src="https://img.shields.io/github/license/Joinn99/RocketEval-ICLR"></a>
<a href="https://colab.research.google.com/github/Joinn99/RocketEval-ICLR/blob/main/Evaluate-LLM-in-Colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

<p align="center" width="100%">
    <kbd><img width="100%" src="https://raw.githubusercontent.com/Joinn99/RepositoryResource/refs/heads/master/RocketEval.svg"> </kbd>
</p>

## 📑 Table of Contents
- [🔧 Installation](#🔧-installation)
- [📦 Download Data](#📦-download-data)
- [🚀 Quick Start](#🚀-quick-start)
- [🤖 Model Support](#🤖-model-support)
- [📝 Preparing Data](#📝-preparing-data)
- [🔄 Running Evaluation Step-by-Step](#🔄-running-evaluation-step-by-step)
   - [Checklist Generation](#checklist-generation)
   - [Checklist Grading](#checklist-grading)
   - [Predicting Scores](#predicting-scores)
   - [Producing Rankings](#producing-rankings)
- [🔄 Simulated Matches for Chatbot Arena](#🔄-simulated-matches-for-chatbot-arena)
- [📚 Reference](#📚-reference)
- [📝 Citation](#📝-citation)

## 🔧 Installation
You can install RocketEval by running the following commands:
```bash
git clone https://github.com/Joinn99/RocketEval-ICLR.git
cd RocketEval-ICLR
pip install -r requirements.txt
```

## 📦 Download Data
The data includes the queries, generated checklists, and responses are stored on [HuggingFace](https://huggingface.co/datasets/Joinn/RocketEval). You can download the data by running the following commands:
```bash
git clone https://huggingface.co/datasets/Joinn/RocketEval && mv RocketEval RocketEval-ICLR/data
```
Alternatively, you can download the data and extract the files manually.
> Notice: Please install [Git LFS](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md) first to clone the data files.

## 🚀 Quick Start
You can start the evaluation on the example `mt-bench` benchmark dataset by running the following commands:

```bash
DATASET=mt-bench
GENERATOR=Deepseek-R1
JUDGE=Qwen2.5-1.5B-Instruct
LABELER=gpt-4o

### API Mode:
export OPENAI_API_KEY=<API_KEY>
export OPENAI_BASE_URL=<URL>
python src/run.py --dataset ${DATASET} --generator ${GENERATOR} --judge ${JUDGE} --train_test --mode api --instant_api --api_parallel_size 16

### Offline Mode:
python src/run.py --dataset ${DATASET} --generator ${GENERATOR} --judge ${JUDGE} --train_test --mode offline --offline_config config/offline/default.yaml
```

## 🤖 Model Support
RocketEval supports two types of model deployments for both checklist generation and grading processes:

### Local Models
- Supports any HuggingFace-compatible models through [vLLM](https://docs.vllm.ai/)
- Configurable through a `yaml` file. Example in `config/offline/default.yaml`:
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct   # Name of the model, can be a local path or a HuggingFace model repo name.
trust_remote_code: true
tensor_parallel_size: 1
gpu_memory_utilization: 0.90
max_model_len: 8192
dtype: auto
seed: 0
max_num_seqs: 128
enable_prefix_caching: true
```
For details, please refer to the [vLLM documentation](https://docs.vllm.ai/en/stable/api/offline_inference/llm.html).

> For checklist grading tasks, we recommend using the local models as they are more stable by introducing `allowed_token_ids` parameter to limit the answers LLM can generate.

### API Models
- Supports OpenAI-compatible APIs, including online API providers like [OpenAI](https://openai.com/api/) and [DeepSeek](https://api-docs.deepseek.com/api/deepseek-api), or local API served by [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) and [SGLang](https://docs.sglang.ai/backend/openai_api_completions.html).
- Two operation modes:
  - Batch mode (recommended)
  - Instant mode

To use API models, you need to configure your API key and base URL in the environment variables:
```bash
export OPENAI_API_KEY=<API_KEY>
export OPENAI_BASE_URL=<URL>
```

## 📝 Preparing Data
We have provided 4 example public benchmark datasets in the `data` folder. 

| Dataset | No. of Queries | Comments |
| --- | --- | --- |
| [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) | 160 | Each 2-turn dialogue is split into 2 queries. |
| [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) | 805 |  |
| [Arena-Hard](https://github.com/lmarena/arena-hard-auto) | 500 |  |
| [WildBench](https://huggingface.co/datasets/allenai/WildBench) | 1,000 | To fit the context window of lightweight LLMs, we use a subset of WildBench including 1000 queries for testing. |

You can also use your own data by preparing the following types of files. All files should be stored using JSON line (.jsonl) format. The data format is mostly following [WildBench](https://huggingface.co/datasets/allenai/WildBench) to ensure compatibility with other evaluation tools.

### Queries

```json
{
    "session_id": "<Identifier of the query in RocketEval>",
    "conversation_input":[
        {"content": "<Historical user query, used as context>", "role":"user"},
        {"content": "<Historical system response, used as context>", "role":"assistant"},
        {"content": "<Current user query>", "role":"user"}
    ],
    "checklist":[],
    "references":{
        "gpt-4": "<Reference response>",
    }
}
```

### Responses

```json
{
    "session_id":"<Identifier of the query in RocketEval>",
    "chat_history":[
        "<Historical user query, used as context>",
        "<Historical system response, used as context>",
        "<Current user query>"
    ],
    "output":["<Reponse to current user query>"],
    "generator":"<Name of generator model>",
}
```

> The fields that exist in [WildBench](https://huggingface.co/datasets/allenai/WildBench) and not used in *RocketEval* are not listed here.

Then put the files in the `data` folder in the following structure:

```
data
├── <DATASET_NAME>
│   ├── queries.jsonl
│   └── response
│       └── <MODEL_NAME_1>.jsonl
│       └── <MODEL_NAME_2>.jsonl
```

All test models stored will be loaded and evaluated by RocketEval automatically. If you want to run evaluation on a specific list of models, you can add `<DATASET_NAME>_train.json` and `<DATASET_NAME>_test.json` in the `config/rankings` folder. The files should contain the list of model names to be included in the training and testing set, respectively. Each element in the JSON file should be:
```json
{
    "name": "<MODEL_NAME>",
    "rating": "<ELO RATING OF MODEL, CONSIDERED AS THE GROUNDTRUTH RANK (OPTIONAL)>"
}
```

## 🔄 Running Evaluation Step-by-Step

Instead of running the evaluation in one command, you can also run the evaluation step-by-step by `src/run_task.py` as follows:

```shell
DATASET=mt-bench
GENERATOR=google/Gemma-2-27B-it
JUDGE=google/Gemma-2-2B-it
LABELER=gpt-4o
# Checklist Generation
python src/run_task.py checklist --dataset ${DATASET} --generator ${GENERATOR}              
# Checklist Grading
python src/run_task.py judgment --dataset ${DATASET} --judge ${JUDGE}                      
# Predicting Scores
python src/run_task.py score --dataset ${DATASET} --judge ${JUDGE} --labeler ${LABELER}     
# Producing Rankings
python src/run_task.py ranking --dataset ${DATASET} --judge ${JUDGE}                       
```

### Checklist Generation

You can generate the checklist by `checklist` option. The function will output the checklist for the test set.
You can modify the `config/template/create.md` to customize the checklist generation prompt.
Alternatively, you can also import the created checklist into a JSON line file. The format of each item is as follows:

```json
{
    "session_id": "<Identifier of the query in RocketEval>",
    "checklist":[
        "<Checklist item 1>",
        "<Checklist item 2>",
        "<Checklist item 3>"
    ]
}
```

### Checklist Grading

Running the `judgment` option will grade the checklist for the specified test models. The function will output the grading results for the test set. The format of each item is as follows:

```json
{
    "session_id": "<Identifier of the query in RocketEval>",
    "model_test": "<Model name>",
    "judge": "<Judge model name>",
    "norm_probability": [0.1, 0.3, 0.5, 0.7, 0.9],
    "judgment": ["No (10%)", "No (30%)", "Unsure (50%)", "Yes (70%)", "Yes (90%)"],
}
```

### Predicting Scores

RocketEval will predict the final scores by learning a predictor from the training set from a powerful judge model (e.g., GPT-4) or directly from humans. To use the score predictor, you need to provide the score for the training set, and specify the labeler model by `--labeler` option. Currently, RocketEval only includes "gpt-4o" as the labeler. You can derive the score from external tools (like [WildBench](https://github.com/allenai/WildBench), [FastChat LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)) and convert the scores to the WildBench compatible format as follows:

```json
{
    "session_id": "<Identifier of the query in RocketEval>",
    "model_test": "<Model name>",
    "score": 3.0
}
```
Then put the score files in the `data` folder in the following structure:

```
data
├── <DATASET_NAME>
│   └── score
│       └── gpt-4o
│           └── <MODEL_NAME_1>.jsonl
│           └── <MODEL_NAME_2>.jsonl
```

### Producing Rankings

You can produce the rankings by `ranking` option. The function will output the rankings for the test set.


## 🔄 Output Simulated Matches for Chatbot Arena

You can output the simulated matches for [LMSYS Chatbot Arena](https://lmarena.ai/) by `chatbot_arena_match` function. The function will output the matches between all test models.

```python
from rocketeval.tools.export import chatbot_arena_match
from rocketeval.data.data_loader import load_target_models

test_model_names = load_target_models(dataset_name="mt-bench", split="test")
result = chatbot_arena_match(dataset_name="mt-bench", judge="Gemma-2-2B-it", model_names=test_model_names)
result.to_json("matches.jsonl", orient="records", lines=True)
```

The output `matches.jsonl` can be loaded by the [notebook](https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH) to calculate the elo rating and conduct analysis.

## 📚 Reference
Here we provide some useful links for the related works in RocketEval.

- LLM Inference
    - [vLLM](https://docs.vllm.ai/en/latest/)
    - [SGLang](https://docs.sglang.ai/)
- LLM Evaluation
    - [LMSYS Chatbot Arena](https://lmarena.ai/)
    - [OpenLLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- Datasets
    - [WildBench](https://github.com/allenai/WildBench)
    - [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
    - [Arena Hard](https://github.com/lmarena/arena-hard-auto)
    - [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)

## 📝 Citation

If you find this work useful in your research, please consider citing the following paper:

```bibtex
@inproceedings{wei2025rocketeval,
    title={RocketEval: Efficient automated {LLM} evaluation via grading checklist},
    author={Tianjun Wei and Wei Wen and Ruizhi Qiao and Xing Sun and Jianghong Ma},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=zJjzNj6QUe}
}
```