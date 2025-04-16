from huggingface_hub import HfApi, create_repo, upload_folder
def upload(number):
    # Define HF Repo name & path
    hf_repo_name = f"wjkim9653/Llama-3-8B-CheckGen-v0-3-{number}"
    local_model_path = f"Models/v0_3/Llama-3-8B-CheckGen-v0-3-{number}"

    # Create Repo
    create_repo(hf_repo_name, exist_ok=True)

    # Upload Model Folder
    upload_folder(
        repo_id=hf_repo_name,
        folder_path=local_model_path,
        commit_message="SFT Trained Llama-3-8B-Instruct based Checklist Generator"
    )

    print(f"✅ Uploaded to https://huggingface.co/{hf_repo_name}")

numbers = [
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

for number in numbers:
    upload(number)