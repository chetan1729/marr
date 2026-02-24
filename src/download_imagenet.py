import os
import subprocess

def download_dataset():
    local_dir = "/mnt/ramdisk/imagenet"
    repo_id = "timm/imagenet-1k-wds"
    
    # 1. Ensure the RAM disk directory exists
    print(f"--- Preparing {local_dir} ---")
    os.makedirs(local_dir, exist_ok=True)
    
    # 2. Build the command
    # We use 32 workers for speed and disable symlinks to force data into RAM
    cmd = [
        "huggingface-cli", "download", repo_id,
        "--repo-type", "dataset",
        "--local-dir", local_dir,
        "--local-dir-use-symlinks", "False",
        "--max-workers", "32"
    ]
    
    print(f"--- Starting Download: {repo_id} ---")
    try:
        subprocess.run(cmd, check=True)
        print("\nDownload Complete!")
    except subprocess.CalledProcessError as e:
        print(f"\nDownload failed with error: {e}")
        return

    # 3. Final Verification
    print("--- Verifying Data Size ---")
    size_cmd = subprocess.run(["du", "-sh", local_dir], capture_output=True, text=True)
    print(f"Total Size on RAM Disk: {size_cmd.stdout.strip()}")
    
    # 4. Check for .tar shards
    train_count = len([f for f in os.listdir(f"{local_dir}/train") if f.endswith('.tar')])
    print(f"Detected {train_count} shards in train folder.")

if __name__ == "__main__":
    download_dataset()