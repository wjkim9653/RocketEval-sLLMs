from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

directory = "v0_3"
checkpoints = [
    "100",
    "1000",
    "2000",
    "3000",
    "4000",
    "5000",
    "6000",
    "7000",
    "8000",
    "9000",
    "10000",
]

def save(directory, checkpoint_number):
    # Load LoRA Adapter Config (just the folder path, not the .json file)
    adapter_path = f"Models/{directory}/checkpoint-{checkpoint_number}"
    peft_config = PeftConfig.from_pretrained(adapter_path)

    # Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, torch_dtype="auto")

    # Load LoRA Adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge LoRA Adapter into Base Model
    merged_model = model.merge_and_unload()

    # Save Merged Model Weights
    merged_model.save_pretrained(f"Models/{directory}/Llama-3-8B-CheckGen-v0-3-{checkpoint_number}")

    # Save Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer.save_pretrained(f"Models/{directory}/Llama-3-8B-CheckGen-v0-3-{checkpoint_number}")

    print(f"✅ Merged model saved to 'Models/{directory}/Llama-3-8B-CheckGen-v0-3-{checkpoint_number}'")

for checkpoint in checkpoints:
    save(directory, checkpoint)