"""
val.py - Jupyter Notebook Version
--------------------------------
Copy these cells into a Jupyter notebook.
Run cells sequentially - model is loaded once and can be reused.
"""

# ============================================================================
# CELL 1: Imports and Config
# ============================================================================
import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import glob

from model.sar_vlm import SARVLM, build_sar_encoder
from dataset import SARVLMDataset, collate_fn


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log(msg: str, log_file: str = None):
    out = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(out)

    if log_file:
        with open(log_file, "a") as f:
            f.write(out + "\n")


def get_latest_checkpoint(checkpoints_dir: str):
    """
    Find the latest checkpoint directory based on step number.
    """
    checkpoint_dirs = glob.glob(os.path.join(checkpoints_dir, "step_*"))
    
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")
    
    # Extract step numbers and sort
    step_dirs = []
    for dir_path in checkpoint_dirs:
        dir_name = os.path.basename(dir_path)
        step_num = int(dir_name.replace("step_", ""))
        step_dirs.append((step_num, dir_path))
    
    step_dirs.sort(key=lambda x: x[0], reverse=True)
    
    latest_step, latest_path = step_dirs[0]
    log(f"Latest checkpoint: {latest_path} (step {latest_step})")
    
    return latest_path, latest_step


# Load config
config = load_config("train_config.yaml")
c_data = config["data"]
c_model = config["model"]
c_lora = config["lora"]
c_train = config["training"]

# Find latest checkpoint
latest_checkpoint_path, global_step = get_latest_checkpoint(c_train["save_dir"])


# ============================================================================
# CELL 2: Device and Tokenizer Setup
# ============================================================================
import gc
import os

# Set environment variable to reduce memory usage
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Running validation on device: {device}")

tokenizer = AutoTokenizer.from_pretrained(c_model["vicuna_path"], use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token


# ============================================================================
# CELL 3: Load Base Model (Run once - takes time)
# ============================================================================
log("Loading SAR Encoder and Hybrid Vicuna Model...")

sar_encoder = build_sar_encoder(
    checkpoint_path=c_model["encoder_checkpoint"],
    freeze=True,
    d_sar=c_model["d_sar"]
)

# Use FP16 to reduce memory usage
vlm = SARVLM.from_vicuna(
    vicuna_path=c_model["vicuna_path"],
    sar_encoder=sar_encoder,
    d_sar=c_model["d_sar"],
    n_visual=c_model["n_visual"],
    lora_r=c_lora["r"],
    lora_alpha=c_lora["alpha"],
    lora_dropout=c_lora["dropout"],
    lora_target_modules=c_lora["target_modules"],
    apply_lora=True,
    torch_dtype=torch.float16  # Use FP16 instead of FP32 to save memory
)

# Enable gradient checkpointing to save memory
vlm.hybrid_vicuna.gradient_checkpointing_enable()

# Move to device
vlm = vlm.to(device)

log("Base model loaded successfully.")


# ============================================================================
# CELL 4: Load Checkpoint Weights (Run this to switch checkpoints)
# ============================================================================
def load_checkpoint(vlm, checkpoint_path, device, log_file=None):
    """Load LoRA adapters and projector from checkpoint."""
    log(f"Loading checkpoint from {checkpoint_path}...", log_file)
    
    # Load LoRA adapters
    vlm.hybrid_vicuna.load_adapter(checkpoint_path, adapter_name="default")
    
    # Load SAR -> Vicuna projector
    projector_path = os.path.join(checkpoint_path, "projector.pth")
    vlm.projector.load_state_dict(torch.load(projector_path, map_location=device))
    
    # Load training state
    training_state_path = os.path.join(checkpoint_path, "training_state.pth")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location=device)
        log(f"Loaded training state: global_step={training_state['global_step']}", log_file)
    
    log("Checkpoint loaded successfully.", log_file)
    return training_state.get('global_step', None) if os.path.exists(training_state_path) else None

# Load the latest checkpoint
global_step = load_checkpoint(vlm, latest_checkpoint_path, device, c_train["log_file"])


# ============================================================================
# CELL 5: Load Validation Dataset
# ============================================================================
log("Loading validation dataset...")

val_dataset = SARVLMDataset(
    c_data["val_jsonl"],
    c_data["data_root"],
    tokenizer,
    max_length=c_train["max_length"]
)

# Use smaller batch size and fewer workers to reduce RAM usage
val_loader = DataLoader(
    val_dataset,
    batch_size=1,  # Reduced from micro_batch_size to save memory
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, tokenizer),
    num_workers=1,  # Reduced from 4 to save RAM
    pin_memory=False  # Disable pin_memory to reduce RAM usage
)

log(f"Validation dataset size: {len(val_dataset)}")


# ============================================================================
# CELL 6: Validation Function Definition
# ============================================================================
# @torch.no_grad()
# def run_validation(vlm, val_loader, device, log_file=None):
#     """Run validation over the complete validation loader."""
#     vlm.eval()

#     total_val_loss = 0.0
#     num_batches = 0

#     val_pbar = tqdm(val_loader, desc="Validation")

#     for batch in val_pbar:
#         # Keep SAR encoder input in FP32
#         sar_input = batch["sar_input"].to(device, dtype=torch.float32)
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
#             outputs = vlm(
#                 sar_input=sar_input,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )

#         loss = outputs.loss
#         total_val_loss += loss.item()
#         num_batches += 1
#         val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

#     if num_batches == 0:
#         return 0.0

#     avg_val_loss = total_val_loss / num_batches
#     log(f"Validation Loss: {avg_val_loss:.4f}", log_file)
#     return avg_val_loss


# # ============================================================================
# # CELL 7: Run Validation (Run this cell to validate with current checkpoint)
# # ============================================================================
# log(f"========== STARTING VALIDATION AT STEP {global_step} ==========", c_train["log_file"])
# avg_val_loss = run_validation(vlm, val_loader, device, c_train["log_file"])
# log(f"========== VALIDATION COMPLETE ==========", c_train["log_file"])
# log(f"Final Validation Loss: {avg_val_loss:.4f}", c_train["log_file"])


# ============================================================================
# CELL 8: Load Different Checkpoint (Optional - to test other checkpoints)
# ============================================================================
# Example: Load a specific checkpoint instead of latest
# checkpoint_path = "/path/to/checkpoints/step_2000"
# global_step = load_checkpoint(vlm, checkpoint_path, device, c_train["log_file"])
# Then re-run CELL 7 to validate with the new checkpoint
# ============================================================================
# CELL 9: DIAGNOSIS — SINGLE SAMPLE LOSS + TEACHER-FORCED PREDICTION
# ============================================================================
#
# PURPOSE:
# --------
# The full validation set currently gives ~0.45 loss.
#
# We want to determine:
#
#   1. What is the loss on ONE specific validation sample?
#   2. What is the sample's question?
#   3. What is its ground-truth answer?
#   4. What does the model predict under teacher forcing?
#   5. What does the model actually generate autoregressively?
#
# IMPORTANT:
# We use the SAME collate_fn and SAME vlm(...) call as validation.
# Therefore this diagnostic follows exactly the validation pathway.
#
# ============================================================================

import math

vlm.eval()

# --------------------------------------------------------------------------
# Select sample
# --------------------------------------------------------------------------

sample_idx = 233

sample = val_dataset[sample_idx]

print("\n" + "=" * 80)
print("SINGLE SAMPLE DIAGNOSTIC")
print("=" * 80)

print(f"Sample index: {sample_idx}")


# --------------------------------------------------------------------------
# Print image information
# --------------------------------------------------------------------------
#
# We need to inspect the actual fields available in the dataset sample.
# This prints all non-tensor fields so we can identify the image ID/path.
# --------------------------------------------------------------------------

print("\n--- SAMPLE FIELDS ---")

for key, value in sample.items():

    if torch.is_tensor(value):
        print(
            f"{key}: tensor "
            f"shape={tuple(value.shape)}, "
            f"dtype={value.dtype}"
        )
    else:
        print(f"{key}: {value}")


# --------------------------------------------------------------------------
# Print question and ground truth
# --------------------------------------------------------------------------

input_ids_single = sample["input_ids"]
labels_single = sample["labels"]

# Full input text
full_text = tokenizer.decode(
    input_ids_single,
    skip_special_tokens=False
)

# Ground-truth answer = labels != -100
valid_label_ids = labels_single[
    labels_single != -100
]

ground_truth = tokenizer.decode(
    valid_label_ids,
    skip_special_tokens=False
)

print("\n--- QUESTION / GROUND TRUTH ---")

print("\nFull input:")
print(repr(full_text))

print("\nGround truth:")
print(repr(ground_truth))


# --------------------------------------------------------------------------
# Reconstruct question only
# --------------------------------------------------------------------------
#
# Everything before the first non--100 label is the prompt.
# --------------------------------------------------------------------------

answer_positions = torch.where(
    labels_single != -100
)[0]

if len(answer_positions) == 0:
    raise RuntimeError(
        "No answer tokens found: all labels are -100."
    )

first_answer_position = answer_positions[0].item()

question_ids = input_ids_single[
    :first_answer_position
]

question_text = tokenizer.decode(
    question_ids,
    skip_special_tokens=False
)

print("\nQuestion/prompt:")
print(repr(question_text))


# ============================================================================
# Create batch EXACTLY like validation
# ============================================================================

batch = collate_fn(
    [sample],
    tokenizer
)

sar_input = batch["sar_input"].to(
    device,
    dtype=torch.float32
)

input_ids = batch["input_ids"].to(device)

attention_mask = batch["attention_mask"].to(device)

labels = batch["labels"].to(device)


print("\n--- BATCH INFORMATION ---")

print("SAR input shape:", tuple(sar_input.shape))
print("Input IDs shape:", tuple(input_ids.shape))
print("Attention mask shape:", tuple(attention_mask.shape))
print("Labels shape:", tuple(labels.shape))

print("\nAttention mask:")
print(attention_mask)


# ============================================================================
# SINGLE SAMPLE FORWARD PASS
# ============================================================================
#
# This is intentionally the same call used by run_validation().
# ============================================================================

print("\n" + "=" * 80)
print("CALCULATING SINGLE SAMPLE LOSS")
print("=" * 80)

with torch.no_grad():

    with torch.cuda.amp.autocast(
        enabled=torch.cuda.is_available()
    ):

        outputs = vlm(
            sar_input=sar_input,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

single_sample_loss = outputs.loss.item()

print("\nSingle sample loss:")
print(f"{single_sample_loss:.6f}")


# ============================================================================
# LOGIT INFORMATION
# ============================================================================

logits = outputs.logits

print("\nLogits shape:")
print(tuple(logits.shape))


# --------------------------------------------------------------------------
# First answer token
# --------------------------------------------------------------------------

first_answer_position = answer_positions[0].item()

ground_truth_first_token_id = labels[
    0,
    first_answer_position
].item()

ground_truth_first_token = tokenizer.decode(
    [ground_truth_first_token_id],
    skip_special_tokens=False
)

print("\n" + "=" * 80)
print("FIRST ANSWER TOKEN")
print("=" * 80)

print("First answer position:", first_answer_position)

print(
    "Ground truth token ID:",
    ground_truth_first_token_id
)

print(
    "Ground truth token:",
    repr(ground_truth_first_token)
)


# --------------------------------------------------------------------------
# Causal LM:
#
# logits at position t predict token t+1.
#
# Therefore the first answer token is predicted using the logits
# immediately before the first answer position.
# --------------------------------------------------------------------------

prediction_position = first_answer_position - 1

first_token_logits = logits[
    0,
    prediction_position
]

first_token_probs = torch.softmax(
    first_token_logits,
    dim=-1
)

ground_truth_probability = first_token_probs[
    ground_truth_first_token_id
].item()

print(
    "\nProbability assigned to ground-truth first token:",
    f"{ground_truth_probability:.8f}"
)

print(
    "Negative log probability:",
    f"{-math.log(max(ground_truth_probability, 1e-12)):.6f}"
)


# ============================================================================
# TOP 20 TEACHER-FORCED PREDICTIONS
# ============================================================================

top_probs, top_ids = torch.topk(
    first_token_probs,
    k=20
)

print("\n" + "=" * 80)
print("TOP 20 TEACHER-FORCED PREDICTIONS")
print("=" * 80)

for rank, (prob, token_id) in enumerate(
    zip(top_probs.tolist(), top_ids.tolist()),
    start=1
):

    token_text = tokenizer.decode(
        [token_id],
        skip_special_tokens=False
    )

    print(
        f"{rank:2d}. "
        f"ID={token_id:6d} "
        f"P={prob:.8f} "
        f"token={repr(token_text)}"
    )


# ============================================================================
# GENERATION
# ============================================================================
#
# IMPORTANT:
# Your SARVLM.generate() returns ONLY the generated tokens.
# Therefore DO NOT do:
#
#   output_ids[0][prompt_length:]
#
# ============================================================================

print("\n" + "=" * 80)
print("AUTOREGRESSIVE GENERATION")
print("=" * 80)

# For generation we use the prompt only.
prompt_ids = input_ids[
    :,
    :first_answer_position
]

prompt_attention_mask = attention_mask[
    :,
    :first_answer_position
]

print("Prompt shape:", tuple(prompt_ids.shape))

with torch.no_grad():

    output_ids = vlm.generate(
        sar_input=sar_input,
        input_ids=prompt_ids,
        attention_mask=prompt_attention_mask,
        max_new_tokens=50,
        do_sample=False
    )

print("\nGenerated token IDs:")
print(output_ids[0].tolist())

generated_text_raw = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=False
)

generated_text = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True
)

print("\nRaw generated text:")
print(repr(generated_text_raw))

print("\nGenerated answer:")
print(repr(generated_text))


# ============================================================================
# FINAL DIAGNOSTIC SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"Sample index:              {sample_idx}")
print(f"Single-sample loss:        {single_sample_loss:.6f}")
print(
    f"Ground-truth first token:  "
    f"{repr(ground_truth_first_token)}"
)
print(
    f"Ground-truth token prob:   "
    f"{ground_truth_probability:.8f}"
)
print(
    f"Generated answer:          "
    f"{repr(generated_text)}"
)

print("=" * 80)