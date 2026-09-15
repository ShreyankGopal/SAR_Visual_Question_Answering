# ==============================================================================
# CELL 1: SETUP AND MODEL LOADING (Run this ONLY ONCE)
# ==============================================================================
import os
import sys
import torch
from PIL import Image
import numpy as np

# Adjust sys path so we can import geochat
# (Make sure your notebook is saved in the GeoChat_Model directory)
sys.path.append(os.path.join(os.getcwd(), 'GeoChat'))

from geochat.conversation import conv_templates, Chat
from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import get_model_name_from_path

print("Initializing model...")
model_path = "MBZUAI/geochat-7B"
model_name = get_model_name_from_path(model_path)

# Load model (this will take a few minutes, but you only do it once per session!)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name, False, False, device="cuda"
)
model = model.eval()

device = "cuda"
chat = Chat(model, image_processor, tokenizer, device=device)
print("Model loaded successfully and is ready in VRAM!")


# ==============================================================================
# CELL 2: EVALUATION LOGIC (Run this as many times as you want)
# ==============================================================================
# You can freely edit and re-run this cell. The model is already loaded in memory!

# 1. Setup a fresh conversation state for this specific image/question
chat_state = conv_templates['llava_v1'].copy()

# 2. Pick an image
test_img_path = "/home/saishruti/Research1/Shreyank_20_credit/Minor_dataset_experiments/patches/sar/TrainArea_1146_p00.tif"
image = Image.open(test_img_path).convert("RGB")

# 3. Upload image to the conversation
img_list = []
chat.upload_img(image, chat_state, img_list)

# 4. Ask a question
user_message = "What is the average SAR response of the Water class in this image?"
print(f"Question: {user_message}")
chat.ask(user_message, chat_state)

# 5. Generate and print the response
print("\nAnswer: ", end="", flush=True)
streamer = chat.stream_answer(
    conv=chat_state, 
    img_list=img_list, 
    temperature=0.6, 
    max_new_tokens=500, 
    max_length=2000
)

output = ""
for new_output in streamer:
    output += new_output
    print(new_output, end="", flush=True)

print("\n\nDone! Feel free to change the question or image path and run this cell again.")
