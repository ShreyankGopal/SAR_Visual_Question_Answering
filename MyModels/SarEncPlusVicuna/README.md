# SAR-VLM: SAR Visual Language Model

A modular SAR visual language model combining frozen SAR encoders with Vicuna LLM through trainable projectors and LoRA adapters.

## Project Structure

```
SarEncPlusVicuna/
├── model/                    # Model components
│   ├── sar_vlm.py          # Main model wrapper (SARVLM)
│   ├── sar_projector.py     # Standard projector
│   ├── sar_ablation_projector.py  # Ablation-specific projector
│   └── hybrid_llama.py      # Hybrid Llama with bidirectional visual attention
├── dataset.py               # PyTorch dataset for SAR-VLM training
├── train.py                 # Main training script (identical to train_benchmark.py)
├── train_benchmark.py       # Training script (same as train.py)
├── train_config.yaml         # Training configuration
├── val.py                   # Validation script
├── Benchmarking/            # Benchmarking and evaluation
│   ├── run_benchmark.py     # Standard benchmarking
│   └── run_benchmark_Ablation_3.py  # Ablation 3 benchmarking
└── tests/                   # Unit tests
```

## Quick Start

```bash
# Training
python train.py --config train_config.yaml

# Validation
python val.py --config train_config.yaml

# Benchmarking
python Benchmarking/run_benchmark.py --config train_config.yaml
```

## Configuration System

All training parameters are controlled via `train_config.yaml`:

```yaml
data:
  train_jsonl: "path/to/train.jsonl"
  val_jsonl: "path/to/val.jsonl"
  data_root: "path/to/data/root"

model:
  vicuna_path: "lmsys/vicuna-7b-v1.5"
  encoder_checkpoint: "path/to/encoder.pth"
  d_sar: 1024                      # SAR encoder output dimension
  n_visual: 256                    # Number of visual tokens

lora:
  r: 16                          # LoRA rank
  alpha: 32                       # LoRA alpha
  dropout: 0.05                   # LoRA dropout
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  epochs: 3
  micro_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  weight_decay: 0.0
  max_length: 512
  save_dir: "path/to/checkpoints"
  log_file: "path/to/training.log"
```

## Adding New Components

### 1. Adding a New Encoder

**Step 1:** Create encoder file in `model/` directory
```python
# model/new_encoder.py
import torch
import torch.nn as nn

class NewEncoder(nn.Module):
    def __init__(self, checkpoint_path, d_output=1024):
        super().__init__()
        self.d_output = d_output
        # Your encoder implementation
        self.backbone = ...
        
    def forward(self, x):
        # Return [B, N_visual, d_output]
        return features
```

**Step 2:** Add encoder factory function in `model/sar_vlm.py`
```python
def build_new_encoder(checkpoint_path, freeze=True, d_sar=1024):
    encoder = NewEncoder(checkpoint_path, d_output=d_sar)
    if freeze:
        encoder.requires_grad_(False)
        encoder.eval()
    return encoder
```

**Step 3:** Update `train_config.yaml` to add encoder selection
```yaml
model:
  encoder_type: "sar"  # or "new_encoder"
  encoder_checkpoint: "path/to/new_encoder.pth"
```

**Step 4:** Update `train.py` model loading section
```python
# In load_config() and main():
if config['model']['encoder_type'] == 'sar':
    sar_encoder = build_sar_encoder(...)
elif config['model']['encoder_type'] == 'new_encoder':
    sar_encoder = build_new_encoder(...)
```

### 2. Adding a New Projector

**Step 1:** Create projector file in `model/` directory
```python
# model/new_projector.py
import torch
import torch.nn as nn

class NewProjector(nn.Module):
    def __init__(self, d_sar=1024, llm_hidden_size=4096, hidden_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_sar, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, llm_hidden_size)
        )
    
    def forward(self, x):
        return self.proj(x)
```

**Step 2:** Update `train_config.yaml`
```yaml
model:
  projector_type: "sar_mlp"  # or "new_projector"
```

**Step 3:** Update `train.py` model loading section
```python
# In main():
if config['model']['projector_type'] == 'sar_mlp':
    from model.sar_ablation_projector import SARAblationProjector
    projector = SARAblationProjector(...)
elif config['model']['projector_type'] == 'new_projector':
    from model.new_projector import NewProjector
    projector = NewProjector(...)
```

### 3. Adding a New Loss Function

**Step 1:** Add loss function to `train.py`
```python
def custom_loss(outputs, labels, alpha=0.5):
    """
    Custom loss combining cross-entropy with additional terms.
    
    Args:
        outputs: Model outputs
        labels: Ground truth labels
        alpha: Weight for additional loss terms
    """
    ce_loss = outputs.loss  # Standard cross-entropy
    
    # Add your custom loss terms here
    custom_term = torch.tensor(0.0, device=outputs.loss.device)
    
    total_loss = ce_loss + alpha * custom_term
    return total_loss
```

**Step 2:** Add loss selection to `train_config.yaml`
```yaml
training:
  loss_type: "cross_entropy"  # or "custom_loss"
  loss_alpha: 0.5  # Custom loss parameters
```

**Step 3:** Update `train.py` training loop
```python
# In train_epoch():
if config['training']['loss_type'] == 'cross_entropy':
    loss = outputs.loss / grad_acc_steps
elif config['training']['loss_type'] == 'custom_loss':
    loss = custom_loss(outputs, labels, config['training']['loss_alpha']) / grad_acc_steps
```

### 4. Modifying Training Logic

**Step 1:** Identify what needs to change in `train.py`:
- Checkpoint frequency: modify `CHECKPOINT_EVERY` constant
- Validation frequency: modify `EVAL_EVERY` constant  
- Optimizer parameters: modify optimizer initialization
- Learning rate schedule: add scheduler in `main()`

**Step 2:** Add corresponding config options to `train_config.yaml`
```yaml
training:
  checkpoint_every: 20          # Save every N batches
  eval_every: 1000               # Validate every N batches
  use_scheduler: true            # Enable learning rate scheduling
  scheduler_type: "cosine"       # Scheduler type
```

**Step 3:** Update `train.py` to use config values
```python
# In main():
CHECKPOINT_EVERY = config['training'].get('checkpoint_every', 20)
EVAL_EVERY = config['training'].get('eval_every', 1000)

if config['training'].get('use_scheduler', False):
    from transformers import get_scheduler
    scheduler = get_scheduler(
        config['training']['scheduler_type'],
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
```

## Training Scripts

### train.py vs train_benchmark.py

Both files are **identical** (they have the same content and file size). They serve the same purpose - main training for the SAR-VLM model. You can use either one; the naming difference might be for different experimental setups but the functionality is the same.

### Key Training Functions

**Main Entry Point:** `main()` in `train.py`
- Loads configuration
- Initializes model, dataset, optimizer
- Runs training loop with checkpointing and validation

**Training Loop:** `train_epoch()`
- Handles forward/backward pass
- Implements gradient accumulation
- Supports mixed precision training

**Checkpointing:** `save_checkpoint()`
- Saves LoRA adapters and projector weights
- Independent of validation/sampling to prevent data loss

**Validation:** `run_validation()`
- Runs evaluation on validation set
- Computes validation loss
- Can be extended for custom metrics

## Model Architecture

### Data Flow
```
SAR Input [B, 1, 512, 512]
    ↓ SAREncoder (frozen)
SAR Features [B, 256, 1024]
    ↓ SARProjector (trainable)
SAR Tokens [B, 256, 4096]
    ↓ Concatenate with text embeddings
Input Embeds [B, 256+N_text, 4096]
    ↓ HybridLlamaForCausalLM (Vicuna + LoRA)
Outputs / Loss
```

### Key Components

**SARVLM** (`model/sar_vlm.py`): Main model wrapper
- Manages encoder, projector, and LLM
- Handles bidirectional visual attention
- Supports both LoRA and non-LoRA modes

**HybridLlamaForCausalLM** (`model/hybrid_llama.py`): Custom Llama model
- Extends standard Llama with bidirectional visual attention
- Supports LoRA adapters for efficient fine-tuning

**SARProjector** (`model/sar_projector.py`): 2-layer MLP projector
- Maps SAR features to LLM embedding space
- Standard architecture for main experiments

**SARAblationProjector** (`model/sar_ablation_projector.py`): Ablation-specific projector
- Same architecture as SARProjector
- Used for Ablation 3 experiments

## Dataset

**SARVLMDataset** (`dataset.py`): PyTorch dataset
- Reads JSONL files with SAR image paths and conversations
- Loads TIFF SAR patches
- Tokenizes conversations in Vicuna v1.5 format
- Supports configurable max sequence length

## Benchmarking

**run_benchmark.py**: Standard benchmarking
- Evaluates model on test/validation sets
- Computes BLEU, ROUGE, Exact Match metrics
- Generates per-sample and summary reports

**run_benchmark_Ablation_3.py**: Ablation 3 benchmarking
- Optimized for memory efficiency
- Supports batch processing
- Includes periodic checkpointing and memory cleanup
- Uses Ablation_3 results prefix

## Configuration Best Practices

1. **Use config flags for conditional logic:**
   ```python
   if config['model']['use_new_feature']:
       # Enable new feature
   ```

2. **Provide sensible defaults:**
   ```python
   CHECKPOINT_EVERY = config['training'].get('checkpoint_every', 20)
   ```

3. **Document new config options:**
   - Add comments in `train_config.yaml`
   - Update this README with new parameters

4. **Keep backward compatibility:**
   - Use `.get()` with defaults for new config options
   - Maintain support for existing experiment setups

## Extension Checklist

When adding new functionality, ensure:

- [ ] Add config option to `train_config.yaml`
- [ ] Update `train.py` to use new config option
- [ ] Add factory function in appropriate model file
- [ ] Update model loading logic in `train.py`
- [ ] Test with config flag both True and False
- [ ] Update this README with new component
- [ ] Add unit tests in `tests/` directory
- [ ] Update validation if needed

## Common Modifications

**Change LLM model:**
1. Update `model.vicuna_path` in config
2. Update `llm_hidden_size` if different from 4096
3. Adjust projector architecture accordingly

**Change batch size:**
1. Update `micro_batch_size` in config
2. Adjust `gradient_accumulation_steps` to maintain effective batch size
3. Monitor GPU memory usage

**Add new metrics:**
1. Add metric computation function in `train.py`
2. Update validation loop to compute new metrics
3. Log metrics to training log file

**Change encoder architecture:**
1. Modify `d_sar` and `n_visual` in config
2. Update projector input dimensions
3. Ensure encoder output matches projector input

## Troubleshooting

**OOM errors:**
- Reduce `micro_batch_size`
- Increase `gradient_accumulation_steps`
- Enable gradient checkpointing

**Poor convergence:**
- Adjust learning rate
- Check gradient flow
- Verify projector architecture matches encoder output

**Slow training:**
- Increase `num_workers` in DataLoader
- Enable pin_memory
- Use mixed precision training