import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F
from dataclasses import dataclass
import time
import math
import os
import numpy as np
import inspect  
from PIL import Image
from tqdm import tqdm
from timm.layers import DropPath
import glob
from torch.utils.data import IterableDataset, DataLoader


torch.backends.cudnn.benchmark = True
##################################################### CONFIG #############################################
## Config 
@dataclass
class ViTConfig:
    # Architecture
    n_layer: int = 12       # 12 blocks for ViT-Base
    n_head: int = 12        # 12 heads for ViT-Base
    n_embd: int = 768       # 768 hidden dim for ViT-Base
    
    # Vision Specifics
    img_size: int = 224     # Input resolution
    patch_size: int = 16    # 16x16 patches
    num_channels: int = 3   # RGB
    num_classes: int = 1000 # ImageNet-1k

    dropout: float = 0.1
    drop_path_rate: float = 0.2 # Stochastic Depth rate
    
    # Derived (for internal use)
    num_patches: int = (224 // 16) ** 2 # 196 + 1 (CLS token) = 197

# -----------------------------------------------------------------------------

class ShardedDataset(IterableDataset):
    def __init__(self, shard_dir, split, ddp_rank, ddp_world_size, shuffle=True):
        super().__init__()
        self.shard_dir = shard_dir
        self.split = split
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.shuffle = shuffle
        
        self.shards = sorted(glob.glob(os.path.join(shard_dir, f"imagenet_{split}_*.npz")))
        if not self.shards:
            raise RuntimeError(f"No shards found for split '{split}' in {shard_dir}")

    def __iter__(self):
        rank = self.ddp_rank
        world_size = self.ddp_world_size
        
        # If we have very few shards, let everyone use all of them for now
        if len(self.shards) < world_size:
            my_shards = self.shards
        else:
            my_shards = [s for i, s in enumerate(self.shards) if i % world_size == rank]
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            my_shards = [s for i, s in enumerate(my_shards) if i % worker_info.num_workers == worker_info.id]

        if not my_shards:
            return

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        while True:
            if self.shuffle:
                np.random.shuffle(my_shards)

            for shard_path in my_shards:
                try:
                    data = np.load(shard_path)
                    # Fixed the ambiguous truth value error here:
                    imgs = data['imgs'] if 'imgs' in data else data['images']
                    labels = data['labels'] if 'labels' in data else data['label']
                    
                    indices = np.arange(len(imgs))
                    if self.shuffle:
                        np.random.shuffle(indices)

                    for idx in indices:
                        img = torch.from_numpy(imgs[idx]).permute(2, 0, 1).float() / 255.0
                        img = (img - mean) / std
                        label = torch.tensor(labels[idx], dtype=torch.long)
                        yield img, label
                except Exception as e:
                    print(f"Rank {rank} Error: {e}")
                    continue
            
            if self.split == 'val':
                break

def create_loader(shard_dir, split, B, ddp_rank, ddp_world_size, num_workers=0):
    ds = ShardedDataset(
        shard_dir=shard_dir, 
        split=split, 
        ddp_rank=ddp_rank, 
        ddp_world_size=ddp_world_size,
        shuffle=(split == 'train')
    )
    
    return DataLoader(
        dataset.shuffle(5000) if split == 'train' else dataset, 
        batch_size=B,
        num_workers=num_workers,
        collate_fn=global_collate_fn, 
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True
    )
##################################################### END DATALOADER #############################################


# Patch Embedding class
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.n_patches = (config.img_size // config.patch_size) ** 2
        
        # 1. Conv2d is used to create the patches and project them to n_embd
        self.patcher = nn.Conv2d(
            in_channels=config.num_channels, 
            out_channels=config.n_embd, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )
        
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
        # 2. Learnable CLS token
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        
        # 3. Learnable Position Embedding (N_patches + 1 for the CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, config.n_embd))
        
        # Initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Create patches: (B, C, H, W) -> (B, n_embd, H/P, W/P)
        x = self.patcher(x) 
        # Flatten: (B, n_embd, H/P, W/P) -> (B, n_embd, n_patches)
        x = self.flatten(x) 
        # Transpose to sequence format: (B, n_patches, n_embd)
        x = x.transpose(1, 2) 
        
        # Prepend CLS token: (B, 1 + n_patches, n_embd)
        cls_tokens = self.class_token.expand(B, -1, -1) # (B, 1, n_embd)
        x = torch.cat((cls_tokens, x), dim=1) 
        
        # Adding Position Embeddings 
        x = x + self.pos_embed # (1, 1 + n_patches, n_embd) broadcasted to (B, 1 + n_patches, n_embd)
        
        return x
    

# MSA class
class MultiheadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # ViT-Base standard uses bias=True for all linear layers
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        # Regularization: ViT usually uses 0.1 dropout for ImageNet-1k
        self.attn_dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)
        self.resid_dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)

    def forward(self, x):
        B, T, C = x.size() # Batch, Sequence (197), Embed (768)

        # 1. Project to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2. Reshape for Multi-Head: (B, T, C) -> (B, nh, T, hs)
        # hs = head_size = C // n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. Scaled Dot Product Attention (Flash Attention)
        # dropout_p is only applied during training
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0, 
            is_causal=False
        )

        # 4. Re-assemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

# MLP class
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. First Linear Layer (Expansion)
        # Standard ViT expansion factor is 4 (e.g., 768 -> 3072) similar to GPT-2's MLP
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        
        # 2. Activation: Exact GELU is standard for ViT
        self.gelu = nn.GELU() 
        
        # 3. Second Linear Layer (Projection back to embedding dim)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
        # 4. Dropout: Critical for training ViT from scratch
        self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, drop_path_rate):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadSelfAttention(config)
        
        # timm's DropPath implementation
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Apply drop_path to the entire branch before adding the residual
        x = x + self.drop_path(self.attn(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Vision-Specific Input Layer
        self.patch_embed = PatchEmbedding(config) 

        # 2. Create the linear decay schedule for DropPath (Stochastic Depth)
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]

        # 3. Transformer Blocks 
        self.blocks = nn.ModuleList([
            Block(config, drop_path_rate=dpr[i]) 
            for i in range(config.n_layer)
        ])

        # 4. Final Normalization & Classification Head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_classes)

        # 5. Weight Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm has both weight and bias
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
        elif isinstance(module, nn.Embedding):
            # Embedding ONLY has weight, NO bias
            torch.nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, x, targets=None):
        # Input x: (B, 3, 224, 224)
        
        # Patching + CLS Token + Positional Encoding
        x = self.patch_embed(x) # (B, 197, n_embd)

        # Transformer Encoder Layers
        # Inside your ViT forward method:
        for block in self.blocks:
            x = block(x)

        # Pre-classification LayerNorm
        x = self.ln_f(x)

        # Classifier: Extract the first token [CLS] for prediction
        # Shape: x[:, 0] is (B, n_embd)
        logits = self.head(x[:, 0]) 

        loss = None
        if targets is not None:
            # Standard CrossEntropy for ImageNet classification
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Filter parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Separate into decay and no-decay groups
        # Anything 2D or higher (weights) decays. 1D (biases/norms) doesn't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Log details on the Master Process (Rank 0)
        if self.config.process_rank == 0:
            num_decay = sum(p.numel() for p in decay_params)
            num_nodecay = sum(p.numel() for p in nodecay_params)
            print(f"Decay params: {len(decay_params)} tensors, {num_decay:,} elements")
            print(f"Non-decay params: {len(nodecay_params)} tensors, {num_nodecay:,} elements")

        # Check for Fused AdamW (Crucial for RTX 50-series speed)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        
        if self.config.process_rank == 0:
            print(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

##################################################### END TRANSFORMER #############################################

##################################################### DDP SETUP #############################################
# set up DDP (Distributed Data Parallelism)
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# 1. Setup DDP
ddp = int(os.environ.get('RANK', -1)) != -1 

if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

# 2. Seed for Reproducibility
# Note: Different ranks usually get different seeds for data shuffling, 
# but the model weights must be synced via DDP.
torch.manual_seed(1337 + ddp_rank) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337 + ddp_rank)

##################################################### DDP SETUP #############################################

##################################################### TRAINING SETUP #############################################
def main():
    total_batch_size = 4096 # Global batch size (images per step)
    B = 128                  # Micro-batch size (images per GPU per forward pass)

    # Calculation:
    # total_batch_size = B * grad_accum_steps * ddp_world_size
    assert total_batch_size % (B * ddp_world_size) == 0, "Total batch size must be divisible by (B * world_size)"
    grad_accum_steps = total_batch_size // (B * ddp_world_size)

    if master_process:
        print(f"Global Batch Size: {total_batch_size}")
        print(f"Micro Batch Size (B): {B}")
        print(f"Number of GPUs: {ddp_world_size}")
        print(f"Gradient Accumulation Steps: {grad_accum_steps}")

    #####################################################################################

    ## DataLoader
    # Initialize Loaders
    train_loader = create_loader("imagenet_shards", "train", B, ddp_rank, ddp_world_size, num_workers=4)
    val_loader = create_loader("imagenet_shards", "val", B, ddp_rank, ddp_world_size, num_workers=4)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # --- LR & Step Calculations ---
    # ImageNet-1k: 1,281,167 images. 
    # Global Batch: 4096. Steps per epoch: ~312
    num_epochs = 300 # 300 epochs to achieve SOTA with ViT-Base on ImageNet-1k dataset
    max_steps = num_epochs * (1281167 // total_batch_size) 
    warmup_steps = 2500 # 10 * (1281167 // total_batch_size) # 10 epochs for warmup

    # Learning rate scaling for large batches
    max_lr = 3e-4  
    min_lr = 1e-5 


    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_sota.txt")
    with open(log_file, "a") as f: 
        pass

    # Initialize Model with Vision Config
    config = ViTConfig()
    config.process_rank = ddp_rank # So the model knows who is master
    model = ViT(config)
    model.to(device)

    # Optimization: Channels Last memory format
    model = model.to(memory_format=torch.channels_last)
    use_compile = True 

    if use_compile:
        model = torch.compile(model, mode="default")

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    # Optimizer with ViT-specific betas
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.3, # ViT needs higher weight decay than GPT
        learning_rate=max_lr, 
        betas=(0.9, 0.95), 
        device_type=device_type
    )

    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # 1. VALIDATION BLOCK
        if step % 250 == 0 or last_step:
            model.eval()
            val_loss_accum = torch.zeros(1, device=device) 
            
            with torch.no_grad():
                for _ in range(val_loss_steps):
                    try:
                        x, y = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        x, y = next(val_iter)
                    
                    x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y) 
                    val_loss_accum += loss.detach()
            
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
            if master_process:
                # Use .item() only when printing/logging
                print(f"step {step:5d} | validation loss: {val_loss_accum.item():.4f}")
                
                # Checkpoint Logic
                if step > 0 and (step % 8000 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': raw_model.config.__dict__,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                    }
                    torch.save(checkpoint, checkpoint_path)

        # 2. TRAINING BLOCK
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum_local = 0.0
        
        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            # Optimization: Ensure x is channels_last and on the correct device
            x = x.to(device, memory_format=torch.channels_last)
            y = y.to(device)
            
            # DDP Grad Sync Optimization
            if ddp:
                # Sync gradients only on the final micro-step
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            
            loss.backward()

        # Global Gradient Clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update Learning Rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize() 
        
        t1 = time.time()
        loss_tensor = torch.tensor(loss_accum_local, device=device)
        if ddp:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        if master_process:
            img_per_sec = total_batch_size / (t1 - t0)
            print(f"step {step:5d} | loss: {loss_tensor.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {(t1-t0)*1000:.2f}ms | img/sec: {img_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()
##################################################### END TRAINING #############################################

if __name__ == "__main__":
    main()