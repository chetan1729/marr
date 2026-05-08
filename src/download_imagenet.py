import os
import subprocess

from huggingface_hub import snapshot_download

def download_dataset():
    train_dir = "/workspace/marr/imagenet_train"
    val_dir = "/dev/shm/imagenet_val"
    repo_id = "timm/imagenet-1k-wds"
    
    # 1. Ensure directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 2. Download Validation Set (to RAM)
    print(f"--- Downloading Validation Set to {val_dir} ---")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=val_dir,
        allow_patterns="imagenet1k-validation-*.tar",
        token=os.getenv("HF_TOKEN")
    )
    
    # 3. Download Training Set (to Disk)
    print(f"--- Downloading Training Set to {train_dir} ---")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=train_dir,
        allow_patterns="imagenet1k-train-*.tar",
        max_workers=32,
        token=os.getenv("HF_TOKEN")
    )

    # 4. Final Verification
    print("--- Verifying Shards ---")
    train_count = len([f for f in os.listdir(train_dir) if f.endswith('.tar')])
    val_count = len([f for f in os.listdir(val_dir) if f.endswith('.tar')])
    print(f"Train Shards: {train_count}")
    print(f"Val Shards: {val_count}")

if __name__ == "__main__":
    download_dataset()