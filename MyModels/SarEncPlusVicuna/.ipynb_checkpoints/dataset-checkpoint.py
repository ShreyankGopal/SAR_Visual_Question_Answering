"""
dataset.py
----------
PyTorch Dataset for SAR-VLM training.
Reads JSONL files, loads TIFF SAR patches, and tokenizes conversations
with Vicuna-1.5 formatting.
 
Supports pulling records from multiple JSONL sources (e.g. mixing in a
second dataset) via the `sources` argument, each with its own data_root
(since image paths are relative to whichever dataset they came from) and
an optional sample_size to cap/subsample that source.
"""
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
try:
    import tifffile
except ImportError:
    raise ImportError("tifffile is required. pip install tifffile")
try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required for non-TIFF sources. pip install Pillow")
 
# Extensions loaded as single-channel TIFFs (no channel duplication assumed).
TIFF_EXTS = (".tif", ".tiff")
# Extensions loaded via PIL where the first channel is kept, on the
# assumption (confirmed for dataset 2) that all channels are duplicates
# of the same single-band SAR data.
RGB_LIKE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
 
 
def load_sar_array(img_path: str):
    """
    Load a SAR patch into a 2D (or already-channel-first) numpy array,
    dispatching on file extension.
 
    TIFFs are read as-is via tifffile. JPEG/PNG/BMP sources are read via
    PIL and, if they come back with multiple channels, only the first
    channel is kept (dataset 2's JPEGs store the same single-band SAR
    data duplicated across 3 channels).
    """
    ext = os.path.splitext(img_path)[1].lower()
 
    if ext in TIFF_EXTS:
        return tifffile.imread(img_path)
 
    if ext in RGB_LIKE_EXTS:
        img_np = np.array(Image.open(img_path))
        if img_np.ndim == 3:
            # Channels are duplicates for this source -- keep the first.
            img_np = img_np[..., 0]
        return img_np
 
    raise ValueError(
        f"Unsupported image format '{ext}' for file: {img_path}"
    )
 
 
def load_jsonl_records(jsonl_path: str):
    """Read a JSONL file into a list of dicts."""
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
 
 
def count_jsonl_records(jsonl_path: str) -> int:
    """Count non-empty lines in a JSONL file without materializing every record."""
    count = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count
 
 
class SARVLMDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str = None,
        data_root: str = None,
        tokenizer=None,
        max_length: int = 512,
        sources: list = None,
        seed: int = 42,
    ):
        """
        Either pass (jsonl_path, data_root) for a single-source dataset
        (original behavior), or pass `sources`, a list of dicts:
            {"jsonl_path": ..., "data_root": ..., "sample_size": int or None}
        Each source is loaded independently and, if sample_size is set and
        smaller than that source's record count, randomly subsampled
        (seeded, so runs are reproducible) before being concatenated in.
        Image paths are resolved against each record's own data_root, so
        sources with different data_roots can be mixed safely.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
 
        if sources is None:
            if jsonl_path is None or data_root is None:
                raise ValueError(
                    "SARVLMDataset requires either (jsonl_path, data_root) "
                    "or `sources`."
                )
            sources = [
                {"jsonl_path": jsonl_path, "data_root": data_root, "sample_size": None}
            ]
 
        rng = random.Random(seed)
 
        # self.records holds (record, data_root, source_idx) tuples so
        # __getitem__ knows which data_root to resolve each record's image
        # path against, and which source it came from (source 0 is always
        # the primary/first dataset -- every other source gets resized to
        # match its spatial size, see target_size below).
        self.records = []
 
        for i, src in enumerate(sources):
            recs = load_jsonl_records(src["jsonl_path"])
            sample_size = src.get("sample_size")
 
            if sample_size is not None and sample_size < len(recs):
                recs = rng.sample(recs, sample_size)
 
            self.records.extend((r, src["data_root"], i) for r in recs)
 
        # If there's more than one source, pin the target spatial size to
        # whatever the first source's own images are, by peeking at its
        # first record. Every non-primary source gets resized to this size
        # in __getitem__ so that collate_fn's torch.stack over a batch
        # never sees mismatched shapes.
        self.target_size = None
 
        if len(sources) > 1 and self.records:
            primary_record, primary_root, _ = next(
                r for r in self.records if r[2] == 0
            )
            primary_img = load_sar_array(
                os.path.join(primary_root, primary_record["image"])
            )
            # load_sar_array always returns a 2D (H, W) array -- TIFFs are
            # single-channel as-is, and RGB-like sources have already been
            # reduced to one channel.
            self.target_size = tuple(primary_img.shape[-2:])
 
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
        # self.system_prompt=SYSTEM_PROMPT
 
    def __len__(self):
        return len(self.records)
 
    def __getitem__(self, idx):
        record, data_root, source_idx = self.records[idx]
 
        # 1. Load SAR Image
        rel_img_path = record["image"]
        abs_img_path = os.path.join(data_root, rel_img_path)
 
        # Load the SAR patch (TIFF or JPEG/PNG/BMP, see load_sar_array),
        # add channel dimension if missing, convert to float32.
        try:
            img_np = load_sar_array(abs_img_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read image at index {idx}: {abs_img_path} "
                f"({type(e).__name__}: {e})"
            ) from e
 
        img_tensor = torch.from_numpy(img_np.copy()).float()
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0) # [1, H, W]
        # Normalize if necessary (assuming 0-255 or 0-65535, we scale to 0-1 as a baseline,
        # or just leave as is if MaRS encoder handles it. We'll leave as is for now.)
        # img_tensor = img_tensor / 255.0
 
        # Resize non-primary sources to match the primary dataset's
        # spatial size, so batches mixing both datasets stack cleanly.
        if self.target_size is not None and source_idx != 0:
            current_size = tuple(img_tensor.shape[-2:])
            if current_size != self.target_size:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),  # -> [1, C, H, W]
                    size=self.target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
 
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
 

# class SARVLMDataset(Dataset):
#     def __init__(self, jsonl_path: str, data_root: str, tokenizer, max_length: int = 512):
#         self.data_root = data_root
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.records = []

#         with open(jsonl_path, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     self.records.append(json.loads(line))

#         # We assume Vicuna v1.5 format:
#         # USER: <question> ASSISTANT: <answer></s>
#         # LAND_COVER_CLASSES = [
#         #     "Bareland", "Rangeland", "Developed Space", "Road",
#         #     "Tree", "Water", "Agriculture Land", "Building",
#         # ]

#         # SYSTEM_PROMPT = (
#         #     "A chat between a curious user and an artificial intelligence assistant. "
#         #     "The assistant gives helpful, detailed, and polite answers to the user's questions. "
#         #     f"The possible land-cover classes are: {', '.join(LAND_COVER_CLASSES)}."
#         # )
#         self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
#         # self.system_prompt=SYSTEM_PROMPT

#     def __len__(self):
#         return len(self.records)

#     def __getitem__(self, idx):
#         record = self.records[idx]

#         # 1. Load SAR Image
#         rel_img_path = record["image"]
#         abs_img_path = os.path.join(self.data_root, rel_img_path)
        
#         # Load TIFF, add channel dimension if missing, convert to float32
#         img_np = tifffile.imread(abs_img_path)
#         img_tensor = torch.from_numpy(img_np).float()
#         if img_tensor.ndim == 2:
#             img_tensor = img_tensor.unsqueeze(0) # [1, H, W]
#         # Normalize if necessary (assuming 0-255 or 0-65535, we scale to 0-1 as a baseline, 
#         # or just leave as is if MaRS encoder handles it. We'll leave as is for now.)
#         # img_tensor = img_tensor / 255.0  

#         # 2. Parse Conversation
#         conv = record["conversations"]
#         human_text = ""
#         gpt_text = ""
#         for turn in conv:
#             if turn["from"] == "human":
#                 human_text = turn["value"]
#                 # Remove <image>\n if present since we prepend visual tokens in the model automatically
#                 human_text = human_text.replace("<image>\n", "").replace("<image>", "")
#             elif turn["from"] == "gpt":
#                 gpt_text = turn["value"]

#         # 3. Format Prompt (Vicuna v1.5 style)
#         prompt = f"{self.system_prompt} USER: {human_text} ASSISTANT:"
#         answer = f" {gpt_text}{self.tokenizer.eos_token}"

#         # 4. Tokenize
#         prompt_tokens = self.tokenizer(
#             prompt, add_special_tokens=True, return_tensors="pt"
#         ).input_ids[0]
        
#         answer_tokens = self.tokenizer(
#             answer, add_special_tokens=False, return_tensors="pt"
#         ).input_ids[0]

#         input_ids = torch.cat([prompt_tokens, answer_tokens])

#         # 5. Create Labels (Mask prompt with -100)
#         labels = torch.cat([
#             torch.full_like(prompt_tokens, -100),
#             answer_tokens
#         ])

#         # Truncate to max length
#         if len(input_ids) > self.max_length:
#             input_ids = input_ids[:self.max_length]
#             labels = labels[:self.max_length]

#         return {
#             "sar_input": img_tensor,
#             "input_ids": input_ids,
#             "labels": labels
#         }

# def collate_fn(batch, tokenizer):
#     """
#     Custom collate_fn to pad input_ids and labels to the max length in the batch.
#     """
#     sar_inputs = torch.stack([item["sar_input"] for item in batch])
    
#     input_ids = [item["input_ids"] for item in batch]
#     labels = [item["labels"] for item in batch]

#     # Pad sequences
#     input_ids_padded = torch.nn.utils.rnn.pad_sequence(
#         input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
#     )
#     labels_padded = torch.nn.utils.rnn.pad_sequence(
#         labels, batch_first=True, padding_value=-100
#     )

#     # Attention mask (1 for real tokens, 0 for pad tokens)
#     attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()

#     return {
#         "sar_input": sar_inputs,
#         "input_ids": input_ids_padded,
#         "attention_mask": attention_mask,
#         "labels": labels_padded
#     }
