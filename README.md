# SAR Visual Question Answering

This project develops a state-of-the-art model that performs Visual Question Answering (VQA) on Synthetic Aperture Radar (SAR) satellite imagery.

## Project Structure

```
Shreyank_20_credit/
├── DataGen/                    # Data generation and preprocessing
│   ├── data/                  # Training/validation data (val.jsonl, test.jsonl)
│   └── reports/               # Dataset statistics and reports
├── GeoChat_Model/             # GeoChat baseline model and benchmarking
│   └── BenchMarks_GeoChat/    # GeoChat benchmark scripts and results
├── MyModels/
│   └── SarEncPlusVicuna/      # Custom SAR-VLM model
│       ├── model/              # Model architecture (SAR encoder, projector, hybrid Vicuna)
│       ├── dataset.py         # SARVLMDataset for loading SAR images and conversations
│       ├── train.py           # Training script
│       ├── val.py             # Validation script (Jupyter notebook format)
│       ├── train_config.yaml  # Training configuration
│       ├── checkpoints/       # Model checkpoints (step_*)
│       └── Benchmarking/      # Benchmarking scripts and results
└── Sar-Chat/                  # Additional SAR chat models
```

## SAR-VLM Model

The custom SAR-VLM model combines:
- **SAR Encoder**: SwinV2-Base based SAR encoder (MaRS)
- **Projector**: MLP to project SAR features to Vicuna embedding space
- **LLM**: Vicuna-7B-v1.5 with LoRA fine-tuning

### Model Architecture
- SAR encoder outputs: `[B, N_visual, d_sar]` where `N_visual=49`, `d_sar=1024`
- Projector maps SAR features to Vicuna hidden size (4096)
- LoRA applied to Vicuna attention layers (rank=8, alpha=16)

## Training

### Setup
```bash
cd MyModels/SarEncPlusVicuna
```

### Configuration
Edit `train_config.yaml` to set:
- Data paths (train.jsonl, val.jsonl, data_root)
- Model paths (vicuna_path, encoder_checkpoint)
- Training parameters (epochs, batch_size, learning_rate)
- LoRA parameters (r, alpha, dropout)

### Run Training
```bash
python train.py
```

Training logs are saved to `training.log`. Checkpoints are saved to `checkpoints/step_*`.

## Validation

### Using val.py (Jupyter Notebook Format)
The `val.py` script is structured as a Jupyter notebook with cells:

1. **Cell 1**: Imports and config loading
2. **Cell 2**: Device and tokenizer setup
3. **Cell 3**: Load base model (run once, takes time)
4. **Cell 4**: Load checkpoint weights (re-run to switch checkpoints)
5. **Cell 5**: Load validation dataset
6. **Cell 6**: Validation function definition
7. **Cell 7**: Run validation
8. **Cell 8**: Load different checkpoint (optional)

### Run Validation
```bash
python val.py
```

Or copy cells into a Jupyter notebook for interactive debugging.

## Benchmarking

### Run Benchmark on Validation Set
```bash
cd MyModels/SarEncPlusVicuna
python Benchmarking/run_benchmark.py --subset 10              # Smoke test
python Benchmarking/run_benchmark.py --category "presence"   # Filter by category
python Benchmarking/run_benchmark.py --checkpoint checkpoints/step_1000  # Specific checkpoint
```

### Metrics
- **Exact Match (EM)**: Case-insensitive match for single-word answers
- **BLEU-1/2/3/4**: Sentence BLEU scores
- **ROUGE-1/2/L**: ROUGE F1 scores

### Outputs
Saved to `Benchmarking/`:
- `results_val_<timestamp>.csv` - Per-sample metrics
- `results_val_<timestamp>.json` - Same data in JSON format
- `summary_val_<timestamp>.txt` - Category-wise and global statistics

## Data Format

The dataset uses JSONL format with conversations:
```json
{
  "id": "lulc_12140_000",
  "image": "patches/sar/TrainArea_2658_p00.tif",
  "category": "regional_vqa",
  "conversations": [
    {"from": "human", "value": "Which land-cover classes are present in the middle right region?"},
    {"from": "gpt", "value": " Rangeland, Developed Space, and Road."}
  ]
}
```

### Land Cover Classes
- Bareland
- Rangeland
- Developed Space
- Road
- Tree
- Water
- Agriculture Land
- Building

### Question Categories
- `regional_vqa`: Questions about land-cover classes in specific regions
- `global_classification`: Binary classification questions
- `region_grounding`: Dominant class in specified regions
- `sar_observation`: SAR response characteristics
- `comparative_spatial`: Comparisons between regions

## Requirements

- Python 3.8+
- PyTorch
- transformers
- peft
- timm
- tifffile
- nltk
- rouge-score
- PyYAML

## Memory Optimization

The model uses FP16 precision and gradient checkpointing to reduce memory usage. For systems with limited RAM:
- Use batch_size=1
- Reduce num_workers
- Disable pin_memory

## License

[Add your license here]
