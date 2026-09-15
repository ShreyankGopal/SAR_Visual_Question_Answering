#!/usr/bin/env python3
"""
validate_projector.py
=====================
Validate the trained MLP projector on validation set.

This script runs automatically after training completes.

Usage:
    python train_benchmark/validate_projector.py --config train_benchmark/config.yaml
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import tifffile
from tqdm import tqdm

# Add GeoChat to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'GeoChat'))

from geochat.model.builder import load_pretrained_model
from geochat.model.multimodal_encoder.sar_encoder import SARVisionTower
from geochat.model.multimodal_projector.builder import build_vision_projector


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class SARVLMDataset(torch.utils.data.Dataset):
    """Dataset for SAR VLM validation with proper causal language modeling labels."""
    
    def __init__(self, data_path, data_root, tokenizer, max_length=512):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load JSONL data
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        # System prompt
        self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load SAR image
        img_path = self.data_root / sample['image']
        img_np = tifffile.imread(img_path)
        img_tensor = torch.from_numpy(img_np).float()
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)  # [1, 512, 512]
        img_tensor = img_tensor.unsqueeze(0)  # [1, 1, 512, 512]
        
        # Get question and answer from conversations
        conversations = sample['conversations']
        human_text = ""
        gpt_text = ""
        for turn in conversations:
            if turn['from'] == 'human':
                human_text = turn['value']
                # Remove <image>\n if present
                human_text = human_text.replace("<image>\n", "").replace("<image>", "")
            elif turn['from'] == 'gpt':
                gpt_text = turn['value']
        
        # Format prompt (Vicuna v1.5 style)
        prompt = f"{self.system_prompt} USER: {human_text} ASSISTANT:"
        answer = f" {gpt_text}{self.tokenizer.eos_token}"
        
        # Tokenize prompt and answer separately
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True, return_tensors='pt').input_ids[0]
        answer_tokens = self.tokenizer(answer, add_special_tokens=False, return_tensors='pt').input_ids[0]
        
        # Concatenate
        input_ids = torch.cat([prompt_tokens, answer_tokens])
        
        # Create labels: mask prompt with -100, keep answer tokens for loss computation
        labels = torch.cat([
            torch.full_like(prompt_tokens, -100),  # -100 = ignore in loss
            answer_tokens
        ])
        
        # Truncate to max length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        return {
            'sar_input': img_tensor,
            'input_ids': input_ids,
            'labels': labels
        }


def collate_fn(batch, tokenizer):
    """Collate function for DataLoader with proper padding."""
    sar_inputs = torch.stack([item['sar_input'] for item in batch])
    
    input_ids_list = [item['input_ids'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    
    # Pad sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )
    
    # Attention mask (1 for real tokens, 0 for pad tokens)
    attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()
    
    return {
        'sar_input': sar_inputs,
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask,
        'labels': labels_padded
    }


@torch.no_grad()
def validate(model, dataloader, device):
    """Run validation with proper autoregressive cross-entropy loss."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc="Validating")
    for batch in pbar:
        sar_input = batch['sar_input'].to(device, dtype=torch.float32)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Use GeoChat's forward method
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=sar_input
            )
            loss = outputs.loss
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    if num_batches == 0:
        return 0.0
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main validation function."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_benchmark/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Setup logging
    log_file = config.get('log_file', 'train_benchmark/train.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_msg(msg):
        out = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(out)
        with open(log_file, 'a') as f:
            f.write(out + '\n')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_msg(f"Using device: {device}")
    
    # Load tokenizer
    log_msg("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['llm_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    # Load GeoChat model
    log_msg("Loading GeoChat model...")
    model_path = config['llm_path']
    model_name = model_path.split('/')[-1]
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, False, False, device=device
    )
    
    # Replace vision tower with SAR encoder if needed
    if config['vision_encoder'] == 'sar':
        log_msg("Replacing vision tower with SAR encoder...")
        sar_encoder_path = os.path.abspath(config['encoder_checkpoint'])
        model.model.vision_tower = SARVisionTower(sar_encoder_path)
        model.model.vision_tower.load_model()
        model.model.vision_tower.to(device=device, dtype=torch.float16)
    
    # Replace projector if needed
    if config['projector_type'] == 'sar_mlp':
        log_msg("Replacing projector with SAR MLP...")
        from geochat.model.multimodal_projector.sar_projector import SARProjector
        d_sar = 1024  # SAR encoder output dim
        llm_hidden_size = 4096  # LLM hidden size
        model.model.mm_projector = SARProjector(d_sar=d_sar, llm_hidden_size=llm_hidden_size)
        model.model.mm_projector.to(device=device, dtype=torch.float16)
    
    # Load checkpoint if specified
    if args.checkpoint:
        log_msg(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.model.mm_projector.load_state_dict(checkpoint['projector_state_dict'])
        log_msg("Checkpoint loaded successfully")
    
    # Freeze LLM and encoder
    if config['llm_frozen']:
        log_msg("Freezing LLM...")
        for name, param in model.named_parameters():
            if 'mm_projector' not in name:
                param.requires_grad = False
    
    if config['encoder_frozen']:
        log_msg("Freezing vision encoder...")
        for param in model.model.vision_tower.parameters():
            param.requires_grad = False
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Load validation dataset
    log_msg("Loading validation dataset...")
    max_length = config.get('max_length', 512)
    dataset = SARVLMDataset(config['validation_data'], config['data_root'], tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=4,
        pin_memory=True
    )
    log_msg(f"Validation dataset size: {len(dataset)}")
    
    # Run validation
    log_msg("Starting validation...")
    avg_loss = validate(model, dataloader, device)
    log_msg(f"Validation completed. Average loss: {avg_loss:.4f}")
    
    # Save validation results
    results = {
        'validation_loss': avg_loss,
        'num_samples': len(dataset),
        'config': config
    }
    
    output_dir = config.get('output_dir', 'train_benchmark/checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'validation_results.json')
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_msg(f"Validation results saved to: {results_path}")


if __name__ == '__main__':
    main()
