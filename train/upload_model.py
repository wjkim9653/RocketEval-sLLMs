from huggingface_hub import HfApi, create_repo, upload_folder

# Define HF Repo name & path
hf_repo_name = "wjkim9653/Llama-3-8B-CheckGen-v0-2"
local_model_path = "Models/v0_2/Llama-3-8B-CheckGen-v0-2"

# Create Repo
create_repo(hf_repo_name, exist_ok=True)

# Upload Model Folder
upload_folder(
    repo_id=hf_repo_name,
    folder_path=local_model_path,
    commit_message="SFT Trained Llama-3-8B-Instruct based Checklist Generator, first try"
)

print(f"✅ Uploaded to https://huggingface.co/{hf_repo_name}")