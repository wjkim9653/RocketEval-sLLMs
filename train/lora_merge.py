from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load LoRA Adapter Config (just the folder path, not the .json file)
adapter_path = "Models/v0_2/checkpoint-1385"
peft_config = PeftConfig.from_pretrained(adapter_path)

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, torch_dtype="auto")

# Load LoRA Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge LoRA Adapter into Base Model
merged_model = model.merge_and_unload()

# Save Merged Model Weights
merged_model.save_pretrained("Models/v0_2/Llama-3-8B-CheckGen-v0-2")

# Save Tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
tokenizer.save_pretrained("Models/v0_2/Llama-3-8B-CheckGen-v0-2")

print("✅ Merged model saved to 'Models/v0_2/Llama-3-8B-CheckGen-v0-2'")