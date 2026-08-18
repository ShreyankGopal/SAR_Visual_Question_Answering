# ==============================================================================
# CELL 1: SETUP AND MODEL LOADING (Run this ONLY ONCE)
# ==============================================================================

import os
import sys
import time
import torch
from PIL import Image
import numpy as np
from datetime import datetime


# ------------------------------------------------------------------------------
# Helper for progress messages
# ------------------------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def timed_step(name, func):
    log(f"⏳ START: {name}")
    start = time.time()

    try:
        result = func()
        elapsed = time.time() - start
        log(f"✅ DONE : {name} ({elapsed:.2f}s)")
        return result

    except Exception as e:
        elapsed = time.time() - start
        log(f"❌ FAILED: {name} after {elapsed:.2f}s")
        log(f"   Error: {type(e).__name__}: {e}")
        raise


# ==============================================================================
# STEP 1 — Basic environment
# ==============================================================================
log("STEP 1/8 — Checking environment...")

log(f"Python executable : {sys.executable}")
log(f"Working directory : {os.getcwd()}")
log(f"PyTorch version   : {torch.__version__}")
log(f"CUDA available    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    log(f"GPU               : {torch.cuda.get_device_name(0)}")
    log(
        f"GPU memory        : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
else:
    raise RuntimeError("CUDA is NOT available.")

log("✅ STEP 1/8 complete")


# ==============================================================================
# STEP 2 — Add GeoChat to Python path
# ==============================================================================
log("STEP 2/8 — Adding GeoChat to sys.path...")

geochat_path = os.path.join(os.getcwd(), "GeoChat")

if not os.path.exists(geochat_path):
    raise FileNotFoundError(
        f"GeoChat directory not found at:\n{geochat_path}"
    )

sys.path.append(geochat_path)

log(f"GeoChat path: {geochat_path}")
log("✅ STEP 2/8 complete")


# ==============================================================================
# STEP 3 — Import GeoChat modules
# ==============================================================================
log("STEP 3/8 — Importing GeoChat modules...")

log("   → Importing conversation...")
from geochat.conversation import conv_templates, Chat
log("   ✓ conversation imported")

log("   → Importing model builder...")
from geochat.model.builder import load_pretrained_model
log("   ✓ model builder imported")

log("   → Importing mm_utils...")
from geochat.mm_utils import get_model_name_from_path
log("   ✓ mm_utils imported")

log("✅ STEP 3/8 complete")


# ==============================================================================
# STEP 4 — Configure model
# ==============================================================================
log("STEP 4/8 — Configuring model...")

model_path = "MBZUAI/geochat-7B"

log(f"Model path: {model_path}")

model_name = timed_step(
    "Determine model name",
    lambda: get_model_name_from_path(model_path)
)

log(f"Detected model name: {model_name}")
log("✅ STEP 4/8 complete")


# ==============================================================================
# STEP 5 — Load pretrained model
# ==============================================================================
log("STEP 5/8 — Loading pretrained GeoChat model...")
log("")
log("⚠️  THIS IS THE SLOW STEP.")
log("    Hugging Face may download/load multiple checkpoint shards.")
log("    Watch the Hugging Face progress bars below.")
log("")

load_start = time.time()

try:
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        False,
        False,
        device="cuda"
    )
except Exception as e:
    log("❌ MODEL LOADING FAILED")
    log(f"Error type: {type(e).__name__}")
    log(f"Error: {e}")
    raise

load_time = time.time() - load_start

log("")
log("STEP 5.5/8 — Fixing corrupted CLIP position embeddings...")
# HuggingFace from_pretrained sees a shape mismatch (577 vs 1297) for the position 
# embeddings in the geochat-7B checkpoint and replaces them with random noise! 
# Since the vision tower is frozen anyway, we just reload the pristine weights 
# from openai/clip-vit-large-patch14-336 and re-interpolate them.
model.get_vision_tower().load_model()
model.get_vision_tower().to(device=model.device, dtype=model.dtype)
log("✅ STEP 5.5/8 complete")

log("")
log(f"✅ Model loading finished in {load_time:.2f}s")
log("✅ STEP 5/8 complete")


# ==============================================================================
# STEP 6 — Put model into evaluation mode
# ==============================================================================
log("STEP 6/8 — Setting model to eval mode...")

model = model.eval()

log("✅ model.eval() complete")
log("✅ STEP 6/8 complete")


# ==============================================================================
# STEP 7 — Create Chat object
# ==============================================================================
log("STEP 7/8 — Creating GeoChat Chat object...")

device = "cuda"

chat = timed_step(
    "Initialize Chat",
    lambda: Chat(
        model,
        image_processor,
        tokenizer,
        device=device
    )
)

log("✅ STEP 7/8 complete")


# ==============================================================================
# STEP 8 — Final GPU diagnostics
# ==============================================================================
log("STEP 8/8 — Checking final GPU state...")

if torch.cuda.is_available():

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    log(f"GPU allocated memory : {allocated:.2f} GB")
    log(f"GPU reserved memory  : {reserved:.2f} GB")

log(f"Context length       : {context_len}")

log("==============================================================")
log("🚀 GEOCHAT IS FULLY LOADED AND READY")
log("==============================================================")

# ==============================================================================
# CELL 2: EVALUATION LOGIC (Run this as many times as you want)
# ==============================================================================
# The model is already loaded in memory from Cell 1. Only Cell 2 needs re-running.

# ── CONFIG — change these freely and re-run ────────────────────────────────────
img_path       = "/home/saishruti/Research1/Shreyank_20_credit/Minor_dataset_experiments/patches/sar/TrainArea_005_p01.tif"
user_message   = "In which part of the image is the bareland more dominant"

# ── Setup fresh conversation state ───────────────────────────────────────────
chat_state = conv_templates['llava_v1'].copy()

# ── Load & upload image ───────────────────────────────────────────────────────
image = Image.open(img_path).convert("RGB")
print(f"Image size : {image.size}")
img_list = []
chat.upload_img(image, chat_state, img_list)

# ── Encode image through CLIP ─────────────────────────────────────────────────
chat.encode_img(img_list)

# ── Ask the question ──────────────────────────────────────────────────────────
print(f"Question   : {user_message}\n")
chat.ask(user_message, chat_state)

# ── Generate response ─────────────────────────────────────────────────────────
print("Generating answer...", flush=True)
model_output = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=256, max_length=2000)

print(f"\nGeoChat Answer: {model_output}")

# ── Visualise the SAR Image and Label Mask ───────────────────────────────────
import sys
import os
# Ensure we can import visualise_image_label
sys.path.append("/home/saishruti/Research1/Shreyank_20_credit/Minor_dataset_experiments")
try:
    from visualise_image_label import visualise
    
    label_path = img_path.replace("patches/sar", "patches/labels")
    if os.path.exists(label_path):
        vis_save_path = "/home/saishruti/Research1/Shreyank_20_credit/GeoChat_Model/test_visualization.png"
        print(f"\nGenerating visualization...")
        visualise(sar_path=img_path, label_path=label_path, save_path=vis_save_path)
    else:
        print(f"\nSkipping visualization, label not found: {label_path}")
except ImportError:
    print("\nCould not import visualise_image_label.py to generate visualization.")

print("\nDone! Change img_path or user_message above and re-run this cell.")
