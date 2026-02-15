# Vision Transformer (ViT-Base) Training on ImageNet-1k

This repository documents the full 300-epoch training run of a **Vision Transformer (ViT-Base)**. The project utilized high-throughput streaming via WebDataset and implemented a heavy-regularization SOTA recipe (Mixup + Stochastic Depth).

## 📊 Final Results (300 Epochs)

The training completed a full cycle of **384.3 million image exposures** over 93,599 steps with a global batch size of 4096.

| Metric | Final Value | status |
| :--- | :--- | :--- |
| **Total Images Seen** | 384,350,100 | 100% Complete |
| **Top-1 Accuracy** | **57.28%** | Final |
| **Top-5 Accuracy** | **80.07%** | Final |
| **Final Val Loss** | **1.85** | Final |

### 🔍 Technical Retrospective
While the model achieved a robust **80.07% Top-5 accuracy**, the Top-1 accuracy finished below the 80% target. 

**Key Findings:**
1. **The "Uncertainty" Bias:** Training with `Mixup=0.8` successfully prevented overfitting but, combined with a conservative Learning Rate, prevented the model from developing high-confidence Top-1 predictions.
2. **Learning Rate Scaling:** The Max LR of `3e-4` was too low for a global batch of `4096`. Following the *Linear Scaling Rule*, a batch of 4096 typically requires a Max LR in the range of `3e-3` to effectively optimize the transformer weights.
3. **The Leading Indicator:** The 80% Top-5 result proves that the model effectively learned the ImageNet feature hierarchy, even if it could not distinguish between fine-grained classes with high Top-1 confidence.



---

## 🛠 Model Configuration

| Hyperparameter | Value |
| :--- | :--- |
| **Architecture** | ViT-Base (12 Layers, 12 Heads, 768 Embd) |
| **Total Steps** | 93,599 |
| **Global Batch Size** | 4096 |
| **Optimizer** | AdamW (Fused) |
| **Regularization** | Mixup ($\alpha=0.8$), Stochastic Depth (0.2), Dropout (0.1) |
| **Augmentation** | RandomResizedCrop, HorizontalFlip, Normalization |

---

## 🚀 Key Features

* **High-Throughput I/O:** Utilizes **WebDataset** for streaming `.tar` shards directly from a Linux ramdisk (`/mnt/ramdisk`).
* **SOTA Augmentation:** Custom `SOTAAug` implementation for large-scale ViT regularization.
* **Distributed Training:** Full **DDP** support with gradient accumulation and `torch.compile` optimization.
* **Memory Efficient:** Channels-last memory format and `bfloat16` autocast for maximum RTX 50-series performance.

---

**📈 Next Steps (V2 Objectives)**

Scale LR: Increase Learning Rate to 3e-3 to match the 4096 batch size.

Scheduler Tuning: Extend the warmup phase to 10k steps to stabilize high-LR training.

Regularization Decay: Implement a schedule to reduce Mixup intensity in the final 50 epochs.

## 💻 Usage

### Requirements
```bash
pip install torch torchvision timm datasets webdataset tqdm```



