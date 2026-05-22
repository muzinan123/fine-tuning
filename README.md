# LLM Fine-Tuning Toolkit — Production-Grade Multi-Model Fine-Tuning Framework

> A systematic fine-tuning engineering practice covering ChatGLM3 & Llama 2, with three training strategies (LoRA / P-Tuning v2 / QLoRA), DeepSpeed ZeRO distributed training, and an interactive Web Demo — designed to bridge the gap between research scripts and production-ready pipelines.

---

## What Problem Does This Solve

Fine-tuning a large language model sounds straightforward — until you hit the real engineering wall:

- Consumer GPUs can't load a 6B+ model in full precision — **how do you fine-tune without a $50k GPU cluster?**
- Different tasks require different adaptation strategies — **how do you switch between LoRA, P-Tuning v2, and QLoRA without rewriting the training pipeline each time?**
- A single training script works on one GPU but breaks on multi-node — **how do you scale from a single RTX 3090 to a 8×A100 cluster with minimal config changes?**
- Models trained in isolation are useless — **how do you go from a fine-tuned checkpoint to a live interactive demo in minutes?**

This project addresses these challenges systematically, with a unified training interface across models and strategies, and a pluggable distributed backend powered by DeepSpeed.

✅ QLoRA (4-bit NF4 quantization) — fine-tune Llama 2 13B on a single 24GB GPU  
✅ Unified Trainer abstraction — swap LoRA / PT2 / QLoRA via config, not code rewrites  
✅ DeepSpeed ZeRO-3 — scale to multi-node with a single config file swap  
✅ Web Demo with conversation history — from checkpoint to interactive UI in one command  

---

## Three Core Engineering Decisions

### Decision 1: QLoRA — Making 13B Fine-Tuning Accessible on a Single GPU

**Problem**: Full fine-tuning of a 13B model requires ~100GB VRAM. Even LoRA in fp16 requires 40GB+. Most practitioners don't have access to A100 clusters.

**Decision**: Implement QLoRA — load the base model in **4-bit NF4 quantization** (bitsandbytes), then attach trainable LoRA adapters in fp16. The base model weights are frozen and quantized; only the adapter parameters (~0.1% of total) are updated.

```
Base Model (4-bit NF4, frozen)
     └── LoRA Adapter (fp16, trainable)
              ├── rank=64, alpha=16
              └── target: q_proj, v_proj, k_proj, o_proj
```

**Outcome**: Llama 2 13B fine-tuning fits within **18GB VRAM** on a single RTX 3090/4090. Training throughput drops ~15% vs fp16 LoRA — an acceptable trade-off for 5× memory reduction.

---

### Decision 2: Unified Trainer Abstraction — One Interface, Three Strategies

**Problem**: ChatGLM3 uses a different tokenization scheme and attention mask convention than Llama 2. P-Tuning v2 requires virtual token injection at the embedding layer. Maintaining three separate training codebases means tripling the maintenance burden.

**Decision**: Build a custom `Trainer` class per model that inherits from HuggingFace `Trainer` and overrides only the strategy-specific logic:

| Component | LoRA | P-Tuning v2 | QLoRA |
|-----------|------|-------------|-------|
| `compute_loss` | Standard CE | Prefix-aware CE | Standard CE |
| `model_init` | PEFT LoraConfig | PrefixTuningConfig | PEFT + BitsAndBytes |
| `data_collator` | Padding mask | Virtual token prepend | Padding mask |
| `save_model` | Adapter weights only | Prefix encoder only | Merged or adapter-only |

**Outcome**: Adding a new fine-tuning strategy requires implementing one method override — the training loop, evaluation, and checkpoint logic are shared across all strategies.

---

### Decision 3: DeepSpeed ZeRO — From Single GPU to Multi-Node with One Config Swap

**Problem**: Scaling from single-GPU to multi-node training typically requires rewriting the training script to handle device placement, gradient synchronization, and optimizer state sharding manually.

**Decision**: Integrate DeepSpeed as the distributed backend. Two pre-configured ZeRO configs cover the full scaling spectrum:

| Config | Sharding | Best For | Memory Saving |
|--------|----------|----------|---------------|
| `ds_config_zero2.json` | Optimizer states + gradients | Single-node multi-GPU | ~4× vs DDP |
| `ds_config_zero3.json` | Optimizer + gradients + parameters | Multi-node / 70B+ models | ~8× vs DDP |

Switching from single-GPU to 8-node training:
```bash
# Single GPU
bash deepspeed/train_on_one_gpu.sh

# Multi-node (change hostfile, same script)
bash deepspeed/train_on_multi_nodes.sh
```

**Outcome**: The same training script runs identically on a single RTX 3090 and a 64-GPU cluster. ZeRO-3 enables training 70B+ models without model parallelism code.

---

## System Overview

```
Raw Data (JSONL / CSV / custom format)
      │
      ▼
Data Pipeline (convert_format.py → combine_and_split.py)
      │
      ▼
┌─────────────────────────────────────┐
│         Training Engine              │
│                                     │
│  Model: ChatGLM3 │ Llama 2          │
│  Strategy: LoRA │ PT2 │ QLoRA       │
│  Backend: Single GPU │ DeepSpeed    │
└─────────────────────────────────────┘
      │
      ▼
Fine-tuned Checkpoint (adapter weights)
      │
      ▼
Web Demo (Gradio UI + conversation history DB)
      │
      ▼
Live Interactive Inference
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# For QLoRA
pip install bitsandbytes>=0.41 peft>=0.6
# For DeepSpeed
pip install deepspeed>=0.12
```

### 2. Prepare Data

```bash
python data/convert_format.py --input raw_data.jsonl --output data/train.json
python data/combine_and_split.py --input data/train.json --split_ratio 0.9
```

### 3. Fine-Tune

```bash
# ChatGLM3 — LoRA
bash chatglm3/lora_train.sh

# ChatGLM3 — P-Tuning v2
bash chatglm3/pt2_train.sh

# Llama 2 — QLoRA (single 24GB GPU)
bash llama2/qlora_train.sh

# DeepSpeed ZeRO-3 multi-node
bash deepspeed/train_on_multi_nodes.sh
```

### 4. Evaluate

```bash
bash chatglm3/lora_eval.sh
bash llama2/qlora_eval.sh
```

### 5. Launch Web Demo

```bash
# ChatGLM3 with LoRA adapter
bash web_demo/chatglm3_lora.sh

# Llama 2 with QLoRA adapter
bash web_demo/llama2_qlora.sh
```

---

## Repository Structure

```
fine-tuning-main/
├── chatglm3/              # ChatGLM3: LoRA & P-Tuning v2
│   ├── trainer.py         # Custom Trainer with PEFT integration
│   ├── main_lora.py       # LoRA entry point
│   ├── main_pt2.py        # P-Tuning v2 entry point
│   └── arguments.py       # Training hyperparameter definitions
├── llama2/                # Llama 2: QLoRA
│   ├── trainer.py         # Custom Trainer with 4-bit quantization
│   ├── main_qlora.py      # QLoRA entry point
│   └── prompt_helper.py   # Prompt template management
├── deepspeed/             # Distributed training
│   ├── config/            # ZeRO-2 / ZeRO-3 configs
│   ├── train_on_one_gpu.sh
│   └── train_on_multi_nodes.sh
├── pretraining/           # GPT-2 pretraining from scratch
├── data/                  # Data format conversion & splitting
├── web_demo/              # Gradio UI + conversation history
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Base Framework | HuggingFace Transformers / PEFT |
| Fine-tuning | LoRA / P-Tuning v2 / QLoRA (bitsandbytes NF4) |
| Distributed | DeepSpeed ZeRO-2 / ZeRO-3 |
| Models | ChatGLM3-6B / Llama 2 7B & 13B |
| Web Demo | Gradio 4.x |
| Data | JSONL / Alpaca / ShareGPT format |
| License | Apache License 2.0 |

---

