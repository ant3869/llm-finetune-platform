# Local LLM Fine-Tuning Platform

![Version](https://img.shields.io/badge/version-0.7.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

A user-friendly platform for fine-tuning local LLMs on consumer hardware (8GB NVIDIA GPU) using QLoRA. Designed for IT automation, ServiceNow training data, and knowledge extraction from books/PDFs.

## Features

- **Memory Efficient**: QLoRA training fits 7B-13B models in 8GB VRAM
- **Multiple Data Formats**: JSON, JSONL, CSV, TXT, PDF, HTML (Confluence)
- **IT Support Templates**: Pre-built formats for ServiceNow tickets, SOPs, Knowledge Articles
- **Book & PDF Templates**: Chapter summaries, Q&A, knowledge extraction, text cleaning
- **HuggingFace Hub Integration**: Browse and load 25+ curated datasets directly
- **Advanced Data Cleaning**: Delimiter removal, pattern matching, deduplication, quality filters
- **Offline Model Support**: Download models at home, use at work without internet
- **Simple CLI**: Command-line training without web UI dependencies
- **Real-time Progress**: Training metrics and VRAM monitoring
- **Modern UI**: Tailwind-inspired styling for a polished experience
- **Contextual Help**: Hover over any control for detailed explanations

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

## Training Templates

Pre-built templates in `data/templates/` organized by category:

### IT Support Templates
- **ServiceNow Tickets** - Ticket analysis and resolution
- **Knowledge Articles** - KB-based Q&A format
- **SOPs & Procedures** - Step-by-step guides
- **Ticket Triage** - Priority & category assignment
- **Incident Postmortems** - Root cause analysis reports
- **Change Requests** - Change management & approvals
- **Customer Communication** - Service incident updates
- **Runbooks** - Alert handling procedures

### Book & PDF Templates (v0.7.0)
- **Chapter Summaries** - Summarize book chapters concisely
- **Book Q&A** - Question answering from book content
- **Knowledge Extraction** - Extract structured knowledge from books
- **PDF Text Cleaning** - Clean and format raw PDF text

## HuggingFace Datasets Browser (v0.7.0)

Browse and load training datasets directly from HuggingFace Hub without leaving the app.

**Features:**
- **8 Curated Categories** - Instruction Following, Code, Q&A, Summarization, Classification, Reasoning, Books, Domain-Specific
- **25+ Popular Datasets** - Dolly, Alpaca, OpenAssistant, CodeAlpaca, SQuAD, and more
- **Dataset Details** - Size, license, description, and use cases for each dataset
- **Field Mapping** - Automatic detection and mapping of dataset fields to training format
- **Custom Datasets** - Load any HuggingFace dataset by ID
- **Seamless Integration** - Loaded data converts to your training format automatically

**Curated Dataset Categories:**
| Category | Examples |
|----------|----------|
| Instruction Following | Dolly 15k, Alpaca, OpenAssistant, UltraChat |
| Code & Programming | CodeAlpaca, CodeSearchNet, Python Alpaca |
| Question Answering | SQuAD, Natural Questions, TriviaQA |
| Summarization | CNN/DailyMail, XSum, SAMSum |
| Text Classification | IMDb, SST-2, AG News |
| Reasoning & Math | GSM8K, MATH, ARC, HellaSwag |
| Books & Literature | BookCorpus, Gutenberg, PG-19 |
| Domain-Specific | Medical QA, Legal QA, Finance Alpaca |

**How to Use:**
1. Go to **Data Preparation** → **🤗 HuggingFace** tab
2. Browse categories or enter a custom dataset ID
3. Select dataset and configure field mapping
4. Click **Load Dataset** to import samples
5. Data is automatically converted to instruction/input/output format

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
- [x] Model comparison dashboard (Milestone 5)
- [x] Hyperparameter optimization (Milestone 5)
- [x] Post-tuning test suite with ratings (Milestone 5)
- [x] Contextual help system (v0.6.0)
- [x] HuggingFace datasets browser (v0.7.0)
- [x] Book & PDF templates (v0.7.0)
- [x] Template categorization (v0.7.0)
- [ ] Training resume & checkpoints (Milestone 6)
- [ ] Multi-GPU support (Milestone 6)

## Model Comparison Dashboard (v0.6.0)

Compare multiple fine-tuned models side-by-side to find the best performer.

**Features:**
- **Training Metrics Comparison** - Compare final loss, training time, and configurations
- **A/B Testing** - Test the same prompt across multiple models
- **Response Quality Comparison** - Side-by-side response comparison with rating
- **Model Rankings** - Automatic ranking based on performance scores
- **Export Reports** - Generate comparison reports in JSON format

**How to Use:**
1. Train multiple models with different configurations
2. Go to **Advanced Tools** → **Model Comparison**
3. Select 2-4 adapters to compare
4. View training metrics or run A/B tests
5. Export comparison report

## Hyperparameter Optimization (v0.6.0)

Automatically find the best training configuration for your model and data.

**Search Methods:**
- **Quick Test** - 2 trials to verify setup (~10 minutes)
- **Random Search** - Sample random configurations
- **Smart Search** - Focus on promising regions (exploitation + exploration)
- **Grid Search** - Exhaustive search (many trials)

**Optimization Presets:**
| Preset | Trials | Focus | Use Case |
|--------|--------|-------|----------|
| Quick Test | 2 | Validation | Verify HPO works |
| Learning Rate Sweep | 6 | Learning rate | Find optimal LR |
| LoRA Optimization | 12 | LoRA params | Optimize r, alpha, dropout |
| Balanced | 10 | Key params | General optimization |
| Full Optimization | 20 | All params | Comprehensive search |

**Tunable Parameters:**
- Learning rate: 1e-5 to 5e-4
- LoRA rank (r): 4, 8, 16, 32, 64
- LoRA alpha: 8, 16, 32, 64, 128
- LoRA dropout: 0.0, 0.05, 0.1
- Batch size: 1, 2, 4
- Gradient accumulation: 4, 8, 16, 32
- Max sequence length: 128, 256, 512, 1024
- Warmup ratio: 0.0, 0.03, 0.1

**How to Use:**
1. Load training data and select a model
2. Go to **Advanced Tools** → **HPO (Auto-Tune)**
3. Choose a preset or customize the search space
4. Start optimization and monitor progress
5. Apply best configuration to training

## Post-Tuning Test Suite (v0.6.0)

Comprehensively evaluate your fine-tuned models with before/after comparison, rating systems, and effectiveness metrics.

**Features:**
- **Before/After Comparison** - Side-by-side responses from base model vs tuned model
- **Test & Rate Interface** - Rate responses on multiple criteria (1-5 scale)
- **Effectiveness Dashboard** - Visual metrics showing improvement by category
- **Test History** - Track all test runs with persistent storage
- **Export Results** - Generate detailed JSON reports

**Rating Criteria:**
| Criterion | Description |
|-----------|-------------|
| Relevance | How relevant is the response to the prompt? |
| Accuracy | Is the information technically correct? |
| Quality | Writing quality, clarity, and structure |
| Helpfulness | Would this response help an IT professional? |
| Overall | Overall impression of the response |

**Built-in IT Support Test Cases:**
- Ticket Summarization - VPN issues, server outages
- Knowledge Article - Cloud migration, security policies  
- Troubleshooting - Email sync, database slow queries
- SOP Generation - Password reset procedures
- Ticket Triage - Priority and category assignment
- Communication - Customer service incident updates
- Technical Analysis - System error log analysis

**Effectiveness Metrics:**
- **Average Improvement Score** - Mean rating increase from base to tuned
- **Category Breakdown** - Performance by test category
- **Response Quality Distribution** - Histogram of ratings
- **Consistency Score** - Variance in response quality

**How to Use:**
1. Train a model and load the adapter
2. Go to **Advanced Tools** → **Post-Tuning Tests**
3. **Before/After Compare** - See side-by-side responses
4. **Test & Rate** - Run test cases and rate responses
5. **Effectiveness Dashboard** - View improvement metrics
6. **Test History** - Review past evaluations

## Contextual Help System (v0.6.0)

The platform includes a subtle, non-intrusive help system that provides contextual information as you navigate.

**How it Works:**
- Hover over any control, label, or button to see an explanation
- Help text appears in the footer at the bottom of the page
- Includes the setting name, brief description, detailed explanation, and tips
- Automatically detects common terms like "epochs", "learning rate", "LoRA rank", etc.

**What's Explained:**
- Training parameters (epochs, batch size, learning rate, sequence length)
- LoRA settings (rank, alpha, dropout, target modules)
- Presets and search methods
- Evaluation metrics (loss, BLEU, ROUGE)
- Navigation steps and advanced tools
- IT support templates

**Coverage:**
- 50+ help entries covering all major features
- Explanations tailored for users new to ML/fine-tuning
- Practical tips and recommendations

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
