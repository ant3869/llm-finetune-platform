# Local LLM Fine-Tuning Platform

A user-friendly platform for fine-tuning local LLMs on consumer hardware (8GB NVIDIA GPU) using QLoRA. Designed for IT automation and ServiceNow training data.

## Features

- **Memory Efficient**: QLoRA training fits 7B-13B models in 8GB VRAM
- **Multiple Data Formats**: JSON, JSONL, CSV, TXT, PDF, HTML (Confluence)
- **IT Support Templates**: Pre-built formats for ServiceNow tickets, SOPs, Knowledge Articles
- **Advanced Data Cleaning**: Delimiter removal, pattern matching, deduplication, quality filters
- **Offline Model Support**: Download models at home, use at work without internet
- **Simple CLI**: Command-line training without web UI dependencies
- **Real-time Progress**: Training metrics and VRAM monitoring
- **Modern UI**: Tailwind-inspired styling for a polished experience

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt

# For Windows, you may need:
pip install bitsandbytes-windows
```

### 2. Prepare Training Data

Place your training data in `data/uploads/` folder. Supported formats:

**JSON/JSONL (Alpaca format - recommended):**
```json
{
  "instruction": "Summarize this ServiceNow ticket",
  "input": "INC0012345: User cannot access VPN...",
  "output": "VPN connectivity issue requiring network troubleshooting."
}
```

See `data/templates/` for IT support examples.

### 3. Run Training

```bash
# Quick test (verify setup works)
python train_cli.py --model microsoft/phi-2 --data ./data/templates/servicenow_ticket.json --preset quick_test

# Balanced training
python train_cli.py --model ibm-granite/granite-3.0-8b-instruct --data ./data/my_training.json --preset balanced

# Custom settings
python train_cli.py --model microsoft/phi-2 --data ./data/data.json --epochs 5 --lr 1e-4 --lora-r 32
```

### 4. Use Your Fine-Tuned Model

The trained LoRA adapter will be saved in `models/adapters/`. You can:
- Load it with the base model for inference
- Merge it into the base model
- Export to GGUF format (coming soon)

## CLI Options

```
Required:
  --model, -m      HuggingFace model ID (e.g., 'microsoft/phi-2')
  --data, -d       Path to training data file

Options:
  --output, -o     Output directory (default: ./models/adapters)
  --preset, -p     Training preset: quick_test, balanced, thorough
  
Training Overrides:
  --epochs         Number of training epochs
  --batch-size     Batch size per device
  --lr             Learning rate
  --max-seq-length Maximum sequence length
  
LoRA Options:
  --lora-r         LoRA rank (default: 16)
  --lora-alpha     LoRA alpha (default: 32)
  
Utility:
  --dry-run        Validate setup without training
  --check-gpu      Show GPU information
  --list-models    List local GGUF models
  --verbose, -v    Verbose output
```

## Training Presets

| Preset | Epochs | Seq Length | Use Case |
|--------|--------|------------|----------|
| quick_test | 1 | 256 | Verify setup, test data format |
| balanced | 3 | 512 | Good quality, reasonable time |
| thorough | 5 | 1024 | Best quality, longer training |

## Memory Requirements

For 8GB VRAM GPUs:
- Use `batch_size: 1` with gradient accumulation
- Enable gradient checkpointing (default)
- Keep max_seq_length ≤ 1024

Estimated VRAM usage for 7B model:
- Model (4-bit): ~2 GB
- LoRA adapters: ~0.2 GB
- Optimizer: ~0.4 GB
- Activations: ~1-3 GB (varies with seq length)
- **Total: ~4-6 GB**

## Project Structure

```
llm-finetune-platform/
├── train_cli.py          # Command-line trainer
├── requirements.txt      # Dependencies
├── config/
│   └── settings.yaml     # Configuration
├── core/
│   ├── model_loader.py   # Model loading utilities
│   ├── dataset_handler.py # Data processing
│   └── trainer.py        # QLoRA training engine
├── data/
│   ├── templates/        # IT support data templates
│   ├── uploads/          # Your training data
│   └── processed/        # Processed datasets
└── models/
    ├── base/             # Local GGUF models
    └── adapters/         # Trained LoRA adapters
```

## Supported Models

Any HuggingFace model compatible with PEFT/LoRA:
- IBM Granite: `ibm-granite/granite-3.0-8b-instruct`
- Microsoft Phi: `microsoft/phi-2`, `microsoft/phi-3-mini-4k-instruct`
- Meta Llama: `meta-llama/Llama-2-7b-hf` (requires access)
- Mistral: `mistralai/Mistral-7B-v0.1`

## Troubleshooting

**CUDA out of memory:**
- Reduce `max_seq_length` (try 256 or 512)
- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps`

**Training loss not decreasing:**
- Check data quality and format
- Try lower learning rate (1e-5)
- Increase training epochs

**bitsandbytes error on Windows:**
- Install Windows-specific version: `pip install bitsandbytes-windows`

## Web UI

Launch the web interface:

```bash
streamlit run app.py
```

The UI provides a guided 5-step workflow:
1. **Data Preparation** - Upload files or use IT support templates (8 templates available)
2. **Model Selection** - Choose from recommended models or use custom
3. **Training** - Configure and monitor training with real-time progress
4. **Evaluation** - Interactive chat testing + batch evaluation with BLEU/ROUGE metrics
5. **Export** - Merge LoRA adapters, export guidance, HuggingFace Hub upload

## IT Support Templates (Milestone 4)

Pre-built templates in `data/templates/`:
- **ServiceNow Tickets** - Ticket analysis and resolution
- **Knowledge Articles** - KB-based Q&A format
- **SOPs & Procedures** - Step-by-step guides
- **Ticket Triage** - Priority & category assignment
- **Incident Postmortems** - Root cause analysis reports
- **Change Requests** - Change management & approvals
- **Customer Communication** - Service incident updates
- **Runbooks** - Alert handling procedures

## Roadmap

- [x] CLI Training (Milestone 1)
- [x] Web UI with Streamlit (Milestone 2)
- [x] Interactive model testing (Milestone 3)
- [x] Model export & GGUF conversion (Milestone 3)
- [x] More IT support templates (Milestone 4)
- [x] Batch inference & evaluation metrics (Milestone 4)
- [x] Advanced data cleaning pipeline (v0.5.0)
- [x] Offline model download support (v0.5.0)
- [x] Tailwind-inspired UI styling (v0.5.0)
- [ ] Model comparison dashboard (Milestone 5)
- [ ] Hyperparameter optimization (Milestone 5)

## Advanced Data Cleaning (v0.5.0)

The data cleaning tab provides powerful text preprocessing:

**Preset Cleaning Modes:**
- **Minimal** - Light cleanup (whitespace, unicode)
- **Standard** - Balanced cleaning for most use cases
- **Aggressive** - Heavy cleaning for messy data
- **IT Support** - Optimized for ticket/KB data

**Features:**
- **Delimiter Removal**: Remove prefixes like `XX |` or `TICKET:` from text
- **Pattern Matching**: Custom regex patterns for targeted removal
- **Quality Filters**: Min/max length, filter empty/duplicates
- **Text Normalization**: Unicode, whitespace, HTML stripping

## Offline Model Support (v0.5.0)

For environments without HuggingFace access (corporate networks):

**Getting Models:**
1. Go to **Model Selection** → **Offline Downloads** tab
2. Click "Get" on a model for download instructions
3. Download at home, transfer via USB to work machine
4. Place files in `./models/base/` (GGUF) or `./models/cache/huggingface/` (HF)

**Supported Offline Models:**
- Phi-2 GGUF (1.8GB) - Great for testing
- Mistral-7B GGUF (4.0GB) - Best quality
- Granite-3B GGUF (1.9GB) - IBM's efficient model
- Llama-2-7B GGUF (3.8GB) - Meta's foundation model

**Environment Variables:**
```bash
# Enable offline mode
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```

## License

MIT License
