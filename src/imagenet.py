import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

def prepare_imagenet_shards(shard_dir="imagenet_shards", images_per_shard=4096, test_mode=False):
    """
    Downloads ImageNet-1k via streaming and saves them into compressed .npz shards.
    Optimized for 2026 using the WebDataset (WDS) stream for instant startup.
    """
    hf_token = os.getenv("HF_TOKEN")

    os.makedirs(shard_dir, exist_ok=True)
    
    # We process validation ficodrst because it's smaller and good for testing
    splits = ['validation', 'train'] 
    
    for split in splits:
        print(f"\n--- Starting {split} split ---")
        
        # Using the WDS version ensures we don't hang at 0it
        # It uses your environment's HF_TOKEN automatically
        try:
            ds = load_dataset(
                "timm/imagenet-1k-wds", 
                split=split, 
                streaming=True,
                token=hf_token
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        shard_idx = 0
        imgs, labels = [], []
        
        # In test_mode, we only take 1000 images to verify the pipeline
        limit = 1000 if test_mode else float('inf')
        
        # Wrap the dataset in a tqdm progress bar
        pbar = tqdm(ds, desc=f"Sharding {split}", unit="img")
        
        for i, sample in enumerate(pbar):
            if i >= limit:
                break
            
            try:
                # 1. Convert to RGB (handles occasional grayscale images)
                # 2. Resize to ViT standard 224x224
                img = sample['jpg'].convert('RGB').resize((224, 224), resample=Image.BICUBIC)
                
                imgs.append(np.array(img, dtype=np.uint8))
                labels.append(sample['cls'])
                
                # If we've hit the shard limit, save to disk
                if (i + 1) % images_per_shard == 0:
                    s_name = 'val' if split == 'validation' else 'train'
                    out_path = os.path.join(shard_dir, f"imagenet_{s_name}_{shard_idx:04d}.npz")
                    
                    np.savez_compressed(out_path, imgs=np.array(imgs), labels=np.array(labels))
                    
                    # Clear memory for the next shard
                    imgs, labels = [], []
                    shard_idx += 1
                    
            except Exception as e:
                # Skip corrupted images if any (rare in WDS)
                continue 

        # Save any remaining images in the final partial shard
        if imgs:
            s_name = 'val' if split == 'validation' else 'train'
            out_path = os.path.join(shard_dir, f"imagenet_{s_name}_{shard_idx:04d}.npz")
            np.savez_compressed(out_path, imgs=np.array(imgs), labels=np.array(labels))
            
        print(f"\nFinished {split}. Created {shard_idx + 1} shards.")

if __name__ == "__main__":
    # Ensure your HF_TOKEN is exported in the terminal before running:
    # export HF_TOKEN="your_token_here"
    
    # Toggle test_mode=False when you are ready for the full 1.2M images
    prepare_imagenet_shards(shard_dir="imagenet_shards", test_mode=False)
