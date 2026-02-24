import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from functools import partial
import os
from tqdm import tqdm

from train import ViT, ViTConfig, global_collate_fn, transform_fn

def run_validation():
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Config & Model Setup
    config = ViTConfig()
    config.process_rank = 0 # Dummy rank for validation
    model = ViT(config)
    
    # 3. Load Checkpoint
    checkpoint_path = "log/model_sota_93599.pt" # Update to your specific step
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle torch.compile/DDP prefixing
        state_dict = checkpoint['model']
        new_state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"Loaded checkpoint: {checkpoint_path} (Step {checkpoint.get('step', 'N/A')})")
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found!")
        return

    model.to(device)
    model.eval() # CRITICAL: Disables Dropout and DropPath

    # 4. Data Loader (Mirroring train.py exactly)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf_op = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    files = "/mnt/ramdisk/imagenet/imagenet1k-validation-*.tar"
    dataset = load_dataset("webdataset", data_files=files, split="train", streaming=True)
    
    # Apply identical transform logic
    dataset = dataset.map(partial(transform_fn, transform_op=tf_op), batched=False)
    
    loader = DataLoader(
        dataset, 
        batch_size=128, 
        num_workers=4, 
        collate_fn=global_collate_fn
    )

    # 5. Evaluation Loop
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    
    print("Starting Validation...")
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            
            # Top-1 and Top-5 Accuracy
            _, pred = logits.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(y.view(1, -1).expand_as(pred))

            top1_correct += correct[:1].reshape(-1).float().sum().item()
            top5_correct += correct[:5].reshape(-1).float().sum().item()
            total_samples += y.size(0)

    # 6. Final Results
    top1_acc = (top1_correct / total_samples) * 100
    top5_acc = (top5_correct / total_samples) * 100
    
    print(f"\n--- Final Results ---")
    print(f"Total Samples: {total_samples}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == "__main__":
    run_validation()