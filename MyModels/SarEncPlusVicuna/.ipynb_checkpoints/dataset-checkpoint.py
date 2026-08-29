"""
dataset.py
----------
PyTorch Dataset for SAR-VLM training.
Reads JSONL files, loads TIFF SAR patches, and tokenizes conversations
with Vicuna-1.5 formatting.
"""
import json
import os
import torch
from torch.utils.data import Dataset
try:
    import tifffile
except ImportError:
    raise ImportError("tifffile is required. pip install tifffile")

class SARVLMDataset(Dataset):
    def __init__(self, jsonl_path: str, data_root: str, tokenizer, max_length: int = 512):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        # We assume Vicuna v1.5 format:
        # USER: <question> ASSISTANT: <answer></s>
        # LAND_COVER_CLASSES = [
        #     "Bareland", "Rangeland", "Developed Space", "Road",
        #     "Tree", "Water", "Agriculture Land", "Building",
        # ]

        # SYSTEM_PROMPT = (
        #     "A chat between a curious user and an artificial intelligence assistant. "
        #     "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        #     f"The possible land-cover classes are: {', '.join(LAND_COVER_CLASSES)}."
        # )
        self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        self.system_prompt=SYSTEM_PROMPT

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # 1. Load SAR Image
        rel_img_path = record["image"]
        abs_img_path = os.path.join(self.data_root, rel_img_path)
        
        # Load TIFF, add channel dimension if missing, convert to float32
        img_np = tifffile.imread(abs_img_path)
        img_tensor = torch.from_numpy(img_np).float()
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0) # [1, H, W]
        # Normalize if necessary (assuming 0-255 or 0-65535, we scale to 0-1 as a baseline, 
        # or just leave as is if MaRS encoder handles it. We'll leave as is for now.)
        # img_tensor = img_tensor / 255.0  

        # 2. Parse Conversation
        conv = record["conversations"]
        human_text = ""
        gpt_text = ""
        for turn in conv:
            if turn["from"] == "human":
                human_text = turn["value"]
                # Remove <image>\n if present since we prepend visual tokens in the model automatically
                human_text = human_text.replace("<image>\n", "").replace("<image>", "")
            elif turn["from"] == "gpt":
                gpt_text = turn["value"]

        # 3. Format Prompt (Vicuna v1.5 style)
        prompt = f"{self.system_prompt} USER: {human_text} ASSISTANT:"
        answer = f" {gpt_text}{self.tokenizer.eos_token}"

        # 4. Tokenize
        prompt_tokens = self.tokenizer(
            prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids[0]
        
        answer_tokens = self.tokenizer(
            answer, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]

        input_ids = torch.cat([prompt_tokens, answer_tokens])

        # 5. Create Labels (Mask prompt with -100)
        labels = torch.cat([
            torch.full_like(prompt_tokens, -100),
            answer_tokens
        ])

        # Truncate to max length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "sar_input": img_tensor,
            "input_ids": input_ids,
            "labels": labels
        }

def collate_fn(batch, tokenizer):
    """
    Custom collate_fn to pad input_ids and labels to the max length in the batch.
    """
    sar_inputs = torch.stack([item["sar_input"] for item in batch])
    
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    # Attention mask (1 for real tokens, 0 for pad tokens)
    attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()

    return {
        "sar_input": sar_inputs,
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded
    }
