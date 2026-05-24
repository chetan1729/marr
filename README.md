# Vision Transformer (ViT-Base) Production Training on ImageNet-1k

A comprehensive engineering log and repository blueprint tracking the complete training trajectory of a custom **Vision Transformer (ViT-Base)** from scratch up to **Step 106,000**. 

This project investigates the synergy between massive data throughput, high-capacity architectures, and aggressive stochastic regularization. It demonstrates a multi-stage training execution on ImageNet-1k utilizing high-throughput streaming pipelines via WebDataset, automated computation graph fusion, and strategic regularization adaptation to scale accuracy from 0% to a definitive **67.01% Top-1 precision**.

---

## 📊 End-to-End Core Metrics Summary

Rather than relying on lenient Top-5 classification metrics, this project evaluates model performance strictly through **Top-1 Accuracy**—forcing single-class alignment and hard feature grounding across all training phases.

### Complete Training Progression
* **Total Image Exposures:** ~434,176,000 samples
* **Global Batch Size:** 4,096 images/step
* **Definitive Exit Step:** 106,000

| Milestone | Phase Focus | Augmentation State | Target Resolution | Top-1 Accuracy | Final Val Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 0** | Initialization | Baseline | $224 \times 224$ | 0.10% (Random) | 6.9080 |
| **Step 93,599** | Phase 1 Climax | `SOTAAug` (Heavy) | $224 \times 224$ | **57.28%** | 1.8500 |
| **Step 105,139** | Phase 2 Climax | Disabled (Stabilization) | $256 \times 256$ | 66.35% | 1.4364 |
| **Step 106,000** | Phase 3 Convergence| Disabled (Stabilization) | $256 \times 256$ | **67.01%** | 1.4347 |

---

## 📈 Consolidated Multi-Phase Engineering Journey

The development trajectory was systematically divided into three major operational phases, each tackling distinct optimization, data pipeline, or architectural bottlenecks.

### 🔹 Phase 1: Foundational Training & Heavy Regularization (Steps 0 – 93,599)
* **Objective:** Establish a rock-solid structural baseline at standard $224 \times 224$ resolution and maximize processing velocity to saturate hardware capacity.
* **Pipeline Infrastructure:** Staged dataset shards in a Linux `tmpfs` (RAMDisk) at `/mnt/ramdisk` combined with a **WebDataset** pipeline to stream data sequentially and completely eliminate random file-system seek latencies.
* **Regularization Loop:** Deployed the heavy `SOTAAug` suite, utilizing a combination of Mixup ($\alpha=0.8$) and CutMix ($\alpha=1.0$) to compensate for the Vision Transformer's lack of CNN-like inductive biases.
* **Outcome:** Successfully completed the foundational 300-epoch training schedule across 93,599 high-velocity steps, yielding a baseline **57.28% Top-1 accuracy** and a validation loss of `1.85`.

### 🔹 Phase 2: Resolution Upscaling & Baseline Fine-Tuning (Steps 93,599 – 105,139)
* **Objective:** Transition the entire structural model pipeline from $224 \times 224$ up to an expanded $256 \times 256$ pixel matrix to allow the attention heads to process more granular spatial details.
* **The Positional Challenge:** Changing resolutions expanded the structural patch sequence length from 197 tokens to 257 tokens ($16 \times 16$ patches + 1 CLS token). A **Bicubic Tensor Interpolation** routine intercepted the pre-trained `pos_embed` weights, smoothly re-mapping the spatial matrices to match the larger input sequence dimension.
* **The Clean Alignment Strategy:** To stabilize the model during this delicate transition, data augmentations were temporarily disabled. Without the distraction of complex augmentations, the model rapidly adapted its spatial maps to the new grid density, reaching a clean-data validation accuracy peak of **66.35%** with a loss of `1.4364` at the phase climax.

### 🔹 Phase 3: High-Resolution Convergence (Steps 105,139 – 106,000)
* **Objective:** Fine-tune the model on the expanded resolution layout to squeeze out maximum feature alignment before finalizing the run.
* **The Mechanics:** The pipeline continued on the clean $256 \times 256$ image distribution to allow the model weights to settle un-distorted on pure ImageNet targets.
* **The Optimization Barrier:** Restarts at this stage initially introduced a validation loss drift up toward `1.4452` due to high optimizer velocity and stale historical momentum states.
* **The Definitive Resolution:** Wiping the stale historical momentum tracks from the AdamW buffers while preserving the core model weights stabilized the optimization loop completely. The model achieved its definitive peak capabilities at **67.01% Top-1 Accuracy** with a highly stable loss profile of `1.4347` at step 106,000, where the project successfully concluded.

---

## 🛠 Model & System Configurations

| Component / Hyperparameter | Operational Setting |
| :--- | :--- |
| **Model Backbone** | ViT-Base (12 Transformer Layers, 12 Attention Heads) |
| **Hidden Embedding Dimension** | 768 Channels |
| **Patch Resolution** | $16 \times 16$ pixels |
| **Global Training Batch Size** | 4,096 Images / Step |
| **Optimizer Slates** | Fused AdamW ($\beta_1=0.9$, $\beta_2=0.999$, Weight Decay: 0.3) |
| **Phase 1 Regularization** | Mixup ($\alpha=0.8$), CutMix ($\alpha=1.0$), Stochastic Depth (0.1), Label Smoothing (0.1) |
| **Phase 2 & 3 Regularization**| Stochastic Depth (Linear schedule up to 0.1), Dropout (0.05), Augmentations Disabled |
| **Loss Vector Metrics** | CrossEntropy with Automated Label Smoothing (0.1) |

---

## 🚀 Advanced Performance Architecture

To scale processing throughput up to **~1,950 images per second** on an NVIDIA GPU cluster, the codebase implements four fundamental PyTorch execution optimizations:

1. **Native FlashAttention:** Utilizes `torch.nn.functional.scaled_dot_product_attention` for fast memory-fused matrix multi-head calculations.
2. **Computational Graph Fusion:** Run with `use_compile = True` via `torch.compile(model, mode="default")` to blend back-to-back transformer layers directly into streamlined CUDA execution kernels.
3. **Channels-Last Memory Layout:** Converts image tensors using `model.to(memory_format=torch.channels_last)` to maximize tensor core computational memory alignment.
4. **Bfloat16 Precision Casting:** Leverages `torch.autocast(dtype=torch.bfloat16)` to optimize operational mixed-precision matrix arithmetic.

---

## 💻 Environment Setup & Local Launch

### 1. Build the Pipeline Environment
Ensure all dependencies are clean, explicitly skipping sandbox build isolation for streaming architectures:
```bash
pip install numpy Cython
pip install streaming --no-build-isolation
pip install torch torchvision timm webdataset datasets tqdm