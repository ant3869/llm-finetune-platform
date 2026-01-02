"""
Help System - Contextual tooltips and explanations.

Provides a non-intrusive help system that displays explanations
in the footer when hovering over UI elements.
"""

import streamlit as st
from typing import Dict, Optional
import json

# ============================================================================
# HELP TEXT DEFINITIONS
# ============================================================================

HELP_TEXTS: Dict[str, Dict[str, str]] = {
    # =========================================================================
    # TRAINING PARAMETERS
    # =========================================================================
    "epochs": {
        "title": "Training Epochs",
        "short": "Number of complete passes through your training data.",
        "detail": "One epoch means the model has seen every training sample once. "
                  "More epochs can improve learning but may cause overfitting. "
                  "Start with 3 epochs for most cases.",
        "tip": "💡 If loss stops decreasing, you've likely trained enough epochs."
    },
    "batch_size": {
        "title": "Batch Size",
        "short": "Number of samples processed together before updating the model.",
        "detail": "Larger batches are more stable but use more VRAM. "
                  "For 8GB GPUs, keep batch size at 1-2 and use gradient accumulation instead.",
        "tip": "💡 Effective batch = batch_size × gradient_accumulation_steps"
    },
    "gradient_accumulation": {
        "title": "Gradient Accumulation Steps",
        "short": "Simulates larger batches without using more VRAM.",
        "detail": "Gradients are accumulated over N steps before updating weights. "
                  "This lets you achieve larger effective batch sizes on limited VRAM. "
                  "Values of 8-16 work well for most cases.",
        "tip": "💡 Higher values = more stable training, but slower updates"
    },
    "learning_rate": {
        "title": "Learning Rate",
        "short": "How much the model adjusts its weights each update.",
        "detail": "Controls the speed of learning. Too high causes instability, "
                  "too low causes slow training. For fine-tuning, 1e-4 to 2e-4 works well. "
                  "LoRA typically uses slightly higher rates than full fine-tuning.",
        "tip": "💡 If training is unstable (loss spikes), try a lower learning rate"
    },
    "max_seq_length": {
        "title": "Maximum Sequence Length",
        "short": "Maximum number of tokens per training sample.",
        "detail": "Longer sequences use more VRAM quadratically. For 8GB GPUs, "
                  "512-1024 is usually safe. Samples longer than this will be truncated.",
        "tip": "💡 Match this to your typical input length + expected output length"
    },
    "warmup_ratio": {
        "title": "Warmup Ratio",
        "short": "Fraction of training used to gradually increase learning rate.",
        "detail": "Starts with a very low learning rate and gradually increases to the target. "
                  "Helps prevent early instability. 0.03-0.1 (3-10%) is typical.",
        "tip": "💡 Use higher warmup (0.1) if training is unstable at start"
    },

    # =========================================================================
    # LORA PARAMETERS
    # =========================================================================
    "lora_r": {
        "title": "LoRA Rank (r)",
        "short": "Dimensionality of the low-rank adaptation matrices.",
        "detail": "Higher rank = more parameters = more capacity to learn, but uses more memory. "
                  "Typical values: 8 (minimal), 16 (balanced), 32-64 (complex tasks). "
                  "Start with 16 and increase if needed.",
        "tip": "💡 r=16 is a good starting point for most fine-tuning tasks"
    },
    "lora_alpha": {
        "title": "LoRA Alpha",
        "short": "Scaling factor for LoRA updates.",
        "detail": "Controls the magnitude of LoRA's influence on the model. "
                  "Usually set to 2× the rank (alpha=32 when r=16). "
                  "Higher alpha = stronger adaptation effect.",
        "tip": "💡 Rule of thumb: alpha = 2 × r (e.g., r=16, alpha=32)"
    },
    "lora_dropout": {
        "title": "LoRA Dropout",
        "short": "Dropout probability for LoRA layers to prevent overfitting.",
        "detail": "Randomly drops connections during training to improve generalization. "
                  "0.0 (no dropout) is fine for small datasets, 0.05-0.1 for larger ones.",
        "tip": "💡 Use 0.05-0.1 if you see signs of overfitting"
    },
    "lora_modules": {
        "title": "Target Modules",
        "short": "Which layers in the model to apply LoRA to.",
        "detail": "LoRA adapts specific attention layers. More modules = more adaptation capacity "
                  "but more memory. Query/Value (q_proj, v_proj) are most common. "
                  "Adding Key and Output can help for complex tasks.",
        "tip": "💡 q_proj and v_proj are usually sufficient"
    },

    # =========================================================================
    # PRESETS & METHODS
    # =========================================================================
    "preset_quick_test": {
        "title": "Quick Test Preset",
        "short": "Minimal training to verify your setup works.",
        "detail": "Runs 1 epoch with 50 max steps to quickly check that data loads, "
                  "model loads, and training starts without errors. Not for real results.",
        "tip": "💡 Always run this first with new data or models"
    },
    "preset_balanced": {
        "title": "Balanced Preset",
        "short": "Good quality results in reasonable time.",
        "detail": "3 epochs with moderate sequence length (512). Good balance between "
                  "training quality and time. Suitable for most use cases.",
        "tip": "💡 Start here for real training after quick test passes"
    },
    "preset_thorough": {
        "title": "Thorough Preset",
        "short": "Best quality but longer training time.",
        "detail": "5 epochs with longer sequences (1024). For when you need the best "
                  "possible results and have time to train. May take several hours.",
        "tip": "💡 Use after you've validated your data and approach"
    },
    "search_random": {
        "title": "Random Search",
        "short": "Randomly samples hyperparameter combinations.",
        "detail": "Simple but effective search strategy. Each trial uses a random "
                  "combination from the search space. Good coverage with enough trials.",
        "tip": "💡 Works well for exploring large search spaces quickly"
    },
    "search_smart": {
        "title": "Smart Search (Bayesian)",
        "short": "Uses past results to guide future trials.",
        "detail": "Builds a probabilistic model of which parameters work well, then "
                  "focuses on promising regions. More efficient than random search "
                  "but needs several initial trials to learn from.",
        "tip": "💡 Best for longer optimization runs (10+ trials)"
    },
    "search_grid": {
        "title": "Grid Search",
        "short": "Exhaustively tests all parameter combinations.",
        "detail": "Tests every combination in the search space. Thorough but can be "
                  "very slow with many parameters. Best for small search spaces.",
        "tip": "💡 Only use with 2-3 parameters, otherwise use random/smart search"
    },

    # =========================================================================
    # HPO PRESETS
    # =========================================================================
    "hpo_quick_test": {
        "title": "Quick Test (HPO)",
        "short": "2 trials to verify HPO setup works.",
        "detail": "Runs just 2 quick trials to make sure hyperparameter optimization "
                  "is working correctly. Not meant to find optimal parameters.",
        "tip": "💡 Run this first to verify your HPO configuration"
    },
    "hpo_lr_sweep": {
        "title": "Learning Rate Sweep",
        "short": "Finds the optimal learning rate for your task.",
        "detail": "Tests 6 different learning rates while keeping other parameters fixed. "
                  "Learning rate is often the most impactful hyperparameter.",
        "tip": "💡 Good first step before optimizing other parameters"
    },
    "hpo_lora_opt": {
        "title": "LoRA Optimization",
        "short": "Optimizes LoRA-specific parameters (rank, alpha, dropout).",
        "detail": "12 trials focused on LoRA parameters. Finds the right balance "
                  "between adaptation capacity and efficiency for your task.",
        "tip": "💡 Use after finding a good learning rate"
    },
    "hpo_balanced": {
        "title": "Balanced HPO",
        "short": "10 trials across key parameters.",
        "detail": "Optimizes learning rate, LoRA rank, and batch size together. "
                  "Good balance of thoroughness and time investment.",
        "tip": "💡 Recommended for most optimization needs"
    },
    "hpo_full": {
        "title": "Full Optimization",
        "short": "20 trials exploring all parameters.",
        "detail": "Comprehensive search across all tunable parameters. Takes longest "
                  "but gives the best chance of finding optimal configuration.",
        "tip": "💡 Use when training time is not a constraint"
    },

    # =========================================================================
    # DATA & MODEL
    # =========================================================================
    "alpaca_format": {
        "title": "Alpaca Format",
        "short": "Standard JSON format for instruction-following training.",
        "detail": "Each sample has: 'instruction' (the task), 'input' (optional context), "
                  "and 'output' (expected response). This is the most common format "
                  "for fine-tuning instruction-following models.",
        "tip": "💡 Most IT support tasks fit well in this format"
    },
    "training_samples": {
        "title": "Training Samples",
        "short": "Your input-output pairs the model learns from.",
        "detail": "Each sample teaches the model one example of the behavior you want. "
                  "Quality matters more than quantity. 100-500 high-quality samples "
                  "often outperform thousands of low-quality ones.",
        "tip": "💡 Review samples for errors before training"
    },
    "validation_split": {
        "title": "Validation Split",
        "short": "Portion of data held out to evaluate training progress.",
        "detail": "Typically 10-20% of your data. Used to check if the model "
                  "generalizes or just memorizes. Validation loss should decrease "
                  "alongside training loss.",
        "tip": "💡 If validation loss increases while training loss decreases, you're overfitting"
    },
    "quantization_4bit": {
        "title": "4-bit Quantization (QLoRA)",
        "short": "Compresses model weights to fit in less VRAM.",
        "detail": "Reduces model size by ~4× by using 4-bit precision instead of 16/32-bit. "
                  "Enables training 7B-13B models on 8GB GPUs with minimal quality loss. "
                  "This platform uses QLoRA by default.",
        "tip": "💡 Essential for consumer GPUs - already enabled by default"
    },
    "base_model": {
        "title": "Base Model",
        "short": "The pre-trained model you're fine-tuning.",
        "detail": "A model already trained on massive amounts of text. Fine-tuning "
                  "adapts it to your specific task. Smaller models (2-3B) train faster, "
                  "larger models (7-8B) often perform better.",
        "tip": "💡 Start with Phi-2 or Phi-3 for testing, Granite/Mistral for production"
    },
    "adapter": {
        "title": "LoRA Adapter",
        "short": "Small add-on weights that customize the base model.",
        "detail": "Instead of modifying the entire model, LoRA trains small adapter "
                  "matrices (~0.1-1% of model size). These can be loaded/unloaded "
                  "easily and combined with different base models.",
        "tip": "💡 Adapters are saved in models/adapters/ after training"
    },

    # =========================================================================
    # EVALUATION
    # =========================================================================
    "loss": {
        "title": "Training Loss",
        "short": "How wrong the model's predictions are during training.",
        "detail": "Measures the difference between predicted and actual outputs. "
                  "Should decrease over time. A sudden spike often indicates "
                  "learning rate issues or data problems.",
        "tip": "💡 Focus on the trend, not individual values"
    },
    "eval_loss": {
        "title": "Validation Loss",
        "short": "Loss measured on held-out validation data.",
        "detail": "Shows how well the model generalizes to unseen data. "
                  "If this increases while training loss decreases, the model "
                  "is memorizing rather than learning (overfitting).",
        "tip": "💡 The gap between train and val loss indicates overfitting"
    },
    "bleu_score": {
        "title": "BLEU Score",
        "short": "Measures n-gram overlap with reference text.",
        "detail": "Originally for machine translation, compares generated text "
                  "to reference outputs. Scores 0-100, higher is better. "
                  "Useful but doesn't capture semantic meaning well.",
        "tip": "💡 Good for consistent output formats, less meaningful for creative tasks"
    },
    "rouge_score": {
        "title": "ROUGE Score",
        "short": "Measures recall of reference content.",
        "detail": "Checks how much of the reference text appears in the generated output. "
                  "ROUGE-1 (single words), ROUGE-2 (word pairs), ROUGE-L (longest sequence). "
                  "Higher is better.",
        "tip": "💡 Good for summarization tasks"
    },

    # =========================================================================
    # UI ACTIONS
    # =========================================================================
    "start_training": {
        "title": "Start Training",
        "short": "Begin the fine-tuning process with current settings.",
        "detail": "Loads the model, prepares data, and starts training. "
                  "Progress will be displayed in real-time. You can stop "
                  "training early if needed - partial adapters are saved.",
        "tip": "💡 Ensure no other GPU-intensive tasks are running"
    },
    "stop_training": {
        "title": "Stop Training",
        "short": "Gracefully stop training and save current progress.",
        "detail": "Stops training at the current step and saves the adapter. "
                  "The partial adapter can still be used, though quality "
                  "depends on how much training completed.",
        "tip": "💡 Training will finish the current step before stopping"
    },
    "export_adapter": {
        "title": "Export Adapter",
        "short": "Save the trained LoRA adapter for use.",
        "detail": "Saves the adapter weights that can be loaded with the base model. "
                  "You can also merge the adapter into the base model to create "
                  "a standalone fine-tuned model.",
        "tip": "💡 Keep adapters small and separate for flexibility"
    },
    "merge_adapter": {
        "title": "Merge Adapter",
        "short": "Combine LoRA adapter with base model permanently.",
        "detail": "Creates a new model with the adapter weights merged in. "
                  "Slightly faster inference but loses flexibility of swapping adapters. "
                  "Required for GGUF conversion.",
        "tip": "💡 Merging is irreversible - keep the original adapter"
    },
    "model_comparison": {
        "title": "Model Comparison",
        "short": "Compare multiple fine-tuned models side by side.",
        "detail": "Load multiple adapters and test them with the same prompts. "
                  "Helps identify which training configuration produced the best results. "
                  "Can export comparison reports.",
        "tip": "💡 Test with diverse prompts representing your use cases"
    },
    "ab_testing": {
        "title": "A/B Testing",
        "short": "Test the same prompt on multiple models.",
        "detail": "Send identical prompts to different adapters and compare "
                  "responses side by side. Useful for qualitative evaluation "
                  "when metrics alone aren't sufficient.",
        "tip": "💡 Use prompts from real-world scenarios for best results"
    },

    # =========================================================================
    # GPU & MEMORY
    # =========================================================================
    "vram_usage": {
        "title": "VRAM Usage",
        "short": "GPU memory currently in use.",
        "detail": "Shows how much of your GPU's video memory is being used. "
                  "Leave headroom for training - if you're at 90%+ before training, "
                  "reduce sequence length or batch size.",
        "tip": "💡 Close other GPU applications (games, video editors) before training"
    },
    "gradient_checkpointing": {
        "title": "Gradient Checkpointing",
        "short": "Trade compute time for memory savings.",
        "detail": "Instead of storing all activations, recomputes them during "
                  "backward pass. Uses ~30% less VRAM but training is ~20% slower. "
                  "Essential for training large models on limited VRAM.",
        "tip": "💡 Enabled by default - don't disable unless you have lots of VRAM"
    },

    # =========================================================================
    # IT SUPPORT TEMPLATES
    # =========================================================================
    "template_servicenow": {
        "title": "ServiceNow Tickets",
        "short": "Training data for ticket analysis and resolution.",
        "detail": "Examples of analyzing ServiceNow tickets and generating "
                  "appropriate responses or solutions. Includes incident classification, "
                  "priority assessment, and resolution suggestions.",
        "tip": "💡 Combine with your own ticket data for best results"
    },
    "template_kb": {
        "title": "Knowledge Articles",
        "short": "Q&A format based on knowledge base content.",
        "detail": "Training examples that answer questions using KB article content. "
                  "Teaches the model to extract and present relevant information "
                  "from technical documentation.",
        "tip": "💡 Great for building an IT support chatbot"
    },
    "template_triage": {
        "title": "Ticket Triage",
        "short": "Priority and category assignment for tickets.",
        "detail": "Examples of analyzing tickets to assign appropriate priority "
                  "levels and categories. Helps automate the initial triage process.",
        "tip": "💡 Customize categories to match your organization"
    },
    "template_runbook": {
        "title": "Runbooks",
        "short": "Alert handling and operational procedures.",
        "detail": "Step-by-step procedures for handling specific alerts or incidents. "
                  "Teaches the model to provide appropriate troubleshooting steps "
                  "for different scenarios.",
        "tip": "💡 Include your actual runbook content for operational relevance"
    },

    # =========================================================================
    # POST-TUNING TESTS
    # =========================================================================
    "test_before_after": {
        "title": "Before/After Comparison",
        "short": "Compare base model vs fine-tuned model responses.",
        "detail": "Shows how the model's responses change after fine-tuning. "
                  "The base model response shows what you started with, "
                  "the tuned response shows what fine-tuning achieved.",
        "tip": "💡 Look for improved relevance, accuracy, and format"
    },
    "test_rating": {
        "title": "Response Rating",
        "short": "Rate model responses on multiple criteria.",
        "detail": "Score responses 1-5 on relevance, accuracy, quality, and helpfulness. "
                  "Builds a quality history over time to track model performance.",
        "tip": "💡 Be consistent in your rating criteria"
    },
    "effectiveness_dashboard": {
        "title": "Effectiveness Dashboard",
        "short": "Visual metrics showing improvement from fine-tuning.",
        "detail": "Aggregates all test ratings to show average improvement scores, "
                  "category breakdowns, and quality distributions. Helps quantify "
                  "the value of fine-tuning.",
        "tip": "💡 Run multiple test cases for meaningful statistics"
    },

    # =========================================================================
    # GENERIC / MISC
    # =========================================================================
    "offline_mode": {
        "title": "Offline Mode",
        "short": "Use pre-downloaded models without internet.",
        "detail": "For environments without HuggingFace access (corporate networks). "
                  "Download models at home, transfer via USB, place in models/base/ "
                  "or models/cache/huggingface/.",
        "tip": "💡 Set HF_HUB_OFFLINE=1 to force offline mode"
    },
    "data_cleaning": {
        "title": "Data Cleaning",
        "short": "Preprocess and clean your training data.",
        "detail": "Remove unwanted patterns, normalize text, filter duplicates, "
                  "and ensure data quality. Clean data leads to better training results.",
        "tip": "💡 Preview changes before applying to catch issues"
    },

    # =========================================================================
    # WORKFLOW NAVIGATION
    # =========================================================================
    "step_data_prep": {
        "title": "Step 1: Data Preparation",
        "short": "Upload and prepare your training data.",
        "detail": "Load data from files (JSON, CSV, TXT, PDF, HTML), use IT support templates, "
                  "or enter samples manually. Preview and clean data before training.",
        "tip": "💡 Quality data matters more than quantity"
    },
    "step_model_select": {
        "title": "Step 2: Model Selection",
        "short": "Choose the base model to fine-tune.",
        "detail": "Select from recommended models or use a custom HuggingFace model. "
                  "Consider model size vs your available VRAM.",
        "tip": "💡 Phi-2 is great for testing, Granite-8B for production"
    },
    "step_training": {
        "title": "Step 3: Training",
        "short": "Configure parameters and run fine-tuning.",
        "detail": "Set training parameters, choose a preset, and monitor training progress. "
                  "Watch VRAM usage and loss curves for issues.",
        "tip": "💡 Start with 'quick_test' to verify your setup"
    },
    "step_evaluation": {
        "title": "Step 4: Evaluation",
        "short": "Test your fine-tuned model.",
        "detail": "Chat with your model interactively, run batch evaluations, "
                  "and measure quality with BLEU/ROUGE metrics.",
        "tip": "💡 Test with prompts similar to your intended use case"
    },
    "step_export": {
        "title": "Step 5: Export",
        "short": "Save and export your fine-tuned model.",
        "detail": "Merge LoRA adapter with base model, export for production use, "
                  "or upload to HuggingFace Hub to share.",
        "tip": "💡 Keep the original adapter for flexibility"
    },
    "tool_model_compare": {
        "title": "Model Comparison",
        "short": "Compare multiple fine-tuned models.",
        "detail": "Load multiple adapters and compare their performance on the same prompts. "
                  "Generate comparison reports to document your findings.",
        "tip": "💡 Test with diverse prompts to find the best model"
    },
    "tool_hpo": {
        "title": "Hyperparameter Optimization",
        "short": "Automatically find optimal training settings.",
        "detail": "Run multiple training trials with different configurations. "
                  "The system finds the best hyperparameters for your data and model.",
        "tip": "💡 Start with a Learning Rate Sweep"
    },
    "tool_post_tuning": {
        "title": "Post-Tuning Tests",
        "short": "Evaluate fine-tuning effectiveness.",
        "detail": "Compare base model vs fine-tuned model responses. Rate quality "
                  "and track improvement metrics over time.",
        "tip": "💡 Document your results for future reference"
    },
}


# ============================================================================
# HELP SYSTEM FUNCTIONS
# ============================================================================

def get_help_text(key: str) -> Optional[Dict[str, str]]:
    """Get help text for a given key."""
    return HELP_TEXTS.get(key)


def format_help_html(key: str) -> str:
    """Format help text as HTML for display."""
    help_info = get_help_text(key)
    if not help_info:
        return ""
    
    title = help_info.get("title", "")
    short = help_info.get("short", "")
    detail = help_info.get("detail", "")
    tip = help_info.get("tip", "")
    
    return f"""
    <div class="help-content">
        <span class="help-title">{title}</span>
        <span class="help-separator">—</span>
        <span class="help-short">{short}</span>
        {f'<span class="help-detail">{detail}</span>' if detail else ''}
        {f'<span class="help-tip">{tip}</span>' if tip else ''}
    </div>
    """


def render_help_footer():
    """Render the help footer with embedded JavaScript that works with Streamlit."""
    # Convert help texts to JSON for JavaScript
    help_texts_json = json.dumps(HELP_TEXTS)
    
    st.markdown(f"""
    <style>
        .help-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, #1a1d24 0%, rgba(26, 29, 36, 0.98) 100%);
            border-top: 1px solid #3d4251;
            padding: 0.6rem 1.5rem 0.6rem calc(1rem + 14rem);
            z-index: 999;
            min-height: 42px;
            display: flex;
            align-items: center;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}
        
        .help-content {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            font-size: 0.8rem;
            line-height: 1.5;
        }}
        
        .help-icon {{
            font-size: 0.9rem;
            opacity: 0.7;
            flex-shrink: 0;
            margin-right: 0.5rem;
        }}
        
        .help-title {{
            font-weight: 600;
            color: #60a5fa;
        }}
        
        .help-separator {{
            color: #6b7280;
            opacity: 0.5;
        }}
        
        .help-short {{
            color: #e0e0e0;
        }}
        
        .help-detail {{
            color: #9ca3af;
            font-size: 0.75rem;
        }}
        
        .help-tip {{
            color: #f59e0b;
            font-size: 0.75rem;
            font-style: italic;
        }}
        
        .help-placeholder {{
            color: #6b7280;
            font-size: 0.75rem;
            font-style: italic;
        }}
        
        .main .block-container {{
            padding-bottom: 4rem !important;
        }}
    </style>
    
    <div class="help-footer" id="help-footer">
        <span class="help-icon">💡</span>
        <span class="help-placeholder">Hover over (?) icons or highlighted terms for detailed explanations</span>
    </div>
    
    <script>
    (function() {{
        const helpTexts = {help_texts_json};
        
        function getHelpFooter() {{
            return document.getElementById('help-footer');
        }}
        
        function updateHelp(key) {{
            const footer = getHelpFooter();
            if (!footer) return;
            
            const help = helpTexts[key];
            if (help) {{
                let html = '<span class="help-icon">💡</span><div class="help-content">';
                html += '<span class="help-title">' + help.title + '</span>';
                html += '<span class="help-separator"> — </span>';
                html += '<span class="help-short">' + help.short + '</span>';
                if (help.detail) {{
                    html += ' <span class="help-detail">' + help.detail + '</span>';
                }}
                if (help.tip) {{
                    html += ' <span class="help-tip">' + help.tip + '</span>';
                }}
                html += '</div>';
                footer.innerHTML = html;
            }}
        }}
        
        function clearHelp() {{
            const footer = getHelpFooter();
            if (!footer) return;
            footer.innerHTML = '<span class="help-icon">💡</span><span class="help-placeholder">Hover over (?) icons or highlighted terms for detailed explanations</span>';
        }}
        
        const labelMappings = {{
            'epochs': 'epochs',
            'batch size': 'batch_size', 
            'gradient accumulation': 'gradient_accumulation',
            'learning rate': 'learning_rate',
            'max sequence length': 'max_seq_length',
            'sequence length': 'max_seq_length',
            'warmup': 'warmup_ratio',
            'lora rank': 'lora_r',
            'lora r': 'lora_r',
            'lora alpha': 'lora_alpha',
            'lora dropout': 'lora_dropout',
            'quick test': 'preset_quick_test',
            'balanced': 'preset_balanced',
            'thorough': 'preset_thorough',
            'random search': 'search_random',
            'smart search': 'search_smart',
            'grid search': 'search_grid',
            'vram': 'vram_usage',
            'training samples': 'training_samples',
            'validation': 'validation_split',
            'loss': 'loss',
            'bleu': 'bleu_score',
            'rouge': 'rouge_score',
            'data cleaning': 'data_cleaning',
            'data preparation': 'step_data_prep',
            'model selection': 'step_model_select',
            'training': 'step_training',
            'evaluation': 'step_evaluation',
            'export': 'step_export',
            'model comparison': 'tool_model_compare',
            'hpo': 'tool_hpo',
            'post-tuning': 'tool_post_tuning',
            'gradient checkpointing': 'gradient_checkpointing',
            'fp16': 'fp16',
            'effective batch': 'batch_size',
            'optimizer': 'learning_rate',
            'adapter': 'adapter',
            'base model': 'base_model'
        }};
        
        const processedElements = new WeakSet();
        
        function setupHelpers() {{
            // Target specific Streamlit elements
            const selectors = [
                'label',
                '[data-testid="stWidgetLabel"]',
                '.stSelectbox label',
                '.stNumberInput label', 
                '.stCheckbox label',
                '.stRadio label',
                '[data-testid="stMetricLabel"]',
                '[data-testid="stMetricValue"]',
                '.stExpander summary span',
                '.stMarkdown h4',
                '.stMarkdown strong'
            ];
            
            document.querySelectorAll(selectors.join(', ')).forEach(el => {{
                if (processedElements.has(el)) return;
                
                const text = (el.textContent || '').toLowerCase().trim();
                if (!text || text.length > 50) return;
                
                for (const [pattern, key] of Object.entries(labelMappings)) {{
                    if (text.includes(pattern)) {{
                        processedElements.add(el);
                        el.style.cursor = 'help';
                        el.setAttribute('title', '');
                        
                        el.addEventListener('mouseenter', function(e) {{
                            e.stopPropagation();
                            updateHelp(key);
                        }});
                        el.addEventListener('mouseleave', function(e) {{
                            e.stopPropagation();
                            clearHelp();
                        }});
                        break;
                    }}
                }}
            }});
            
            // Also handle Streamlit's built-in help (?) tooltips - show in footer too
            document.querySelectorAll('[data-testid="stTooltipIcon"]').forEach(icon => {{
                if (processedElements.has(icon)) return;
                processedElements.add(icon);
                
                const parent = icon.closest('[data-testid="stWidgetLabel"]') || icon.parentElement;
                if (parent) {{
                    const labelText = (parent.textContent || '').toLowerCase();
                    for (const [pattern, key] of Object.entries(labelMappings)) {{
                        if (labelText.includes(pattern)) {{
                            icon.addEventListener('mouseenter', () => updateHelp(key));
                            icon.addEventListener('mouseleave', clearHelp);
                            break;
                        }}
                    }}
                }}
            }});
        }}
        
        // Run setup with delay for Streamlit rendering
        function init() {{
            setTimeout(setupHelpers, 300);
        }}
        
        // Initial setup
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', init);
        }} else {{
            init();
        }}
        
        // Re-run on mutations (Streamlit updates)
        let debounceTimer;
        const observer = new MutationObserver(() => {{
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(setupHelpers, 200);
        }});
        
        observer.observe(document.body, {{ 
            childList: true, 
            subtree: true,
            attributes: false
        }});
    }})();
    </script>
    """, unsafe_allow_html=True)


def inject_help_system_js():
    """Inject JavaScript for hover detection and help display."""
    # Convert help texts to JSON for JavaScript
    help_texts_json = json.dumps(HELP_TEXTS)
    
    st.markdown(f"""
    <script>
    (function() {{
        // Help texts dictionary
        const helpTexts = {help_texts_json};
        
        // Debounce helper
        let initTimeout = null;
        
        // Get or create help footer
        function getHelpFooter() {{
            let footer = document.getElementById('help-footer');
            if (!footer) {{
                footer = document.querySelector('.help-footer');
            }}
            return footer;
        }}
        
        // Update help footer content
        function updateHelp(key) {{
            const footer = getHelpFooter();
            if (!footer) return;
            
            const help = helpTexts[key];
            if (help) {{
                footer.classList.add('has-content');
                let html = '<span class="help-icon">💡</span><div class="help-content">';
                html += '<span class="help-title">' + help.title + '</span>';
                html += '<span class="help-separator">—</span>';
                html += '<span class="help-short">' + help.short + '</span>';
                if (help.detail) {{
                    html += '<span class="help-detail">' + help.detail + '</span>';
                }}
                if (help.tip) {{
                    html += '<span class="help-tip">' + help.tip + '</span>';
                }}
                html += '</div>';
                footer.innerHTML = html;
            }}
        }}
        
        // Clear help footer
        function clearHelp() {{
            const footer = getHelpFooter();
            if (!footer) return;
            footer.classList.remove('has-content');
            footer.innerHTML = '<span class="help-icon">💡</span><span class="help-placeholder">Hover over any control for more information</span>';
        }}
        
        // Map of label text patterns to help keys
        const labelMappings = {{
            // Training parameters
            'epochs': 'epochs',
            'batch size': 'batch_size',
            'gradient accumulation': 'gradient_accumulation',
            'learning rate': 'learning_rate',
            'max sequence length': 'max_seq_length',
            'max seq': 'max_seq_length',
            'sequence length': 'max_seq_length',
            'warmup ratio': 'warmup_ratio',
            'warmup': 'warmup_ratio',
            
            // LoRA parameters
            'lora rank': 'lora_r',
            'lora r': 'lora_r',
            'rank (r)': 'lora_r',
            'lora alpha': 'lora_alpha',
            'lora dropout': 'lora_dropout',
            'target modules': 'lora_modules',
            
            // Presets
            'training preset': 'preset_balanced',
            'quick_test': 'preset_quick_test',
            'quick test': 'preset_quick_test',
            'balanced': 'preset_balanced',
            'thorough': 'preset_thorough',
            
            // Search methods
            'random search': 'search_random',
            'smart search': 'search_smart',
            'grid search': 'search_grid',
            'search method': 'search_random',
            
            // HPO presets
            'learning rate sweep': 'hpo_lr_sweep',
            'lora optimization': 'hpo_lora_opt',
            'full optimization': 'hpo_full',
            'number of trials': 'hpo_balanced',
            
            // Model & Data
            'vram': 'vram_usage',
            'gpu status': 'vram_usage',
            'training samples': 'training_samples',
            'samples loaded': 'training_samples',
            'validation': 'validation_split',
            'base model': 'base_model',
            'adapter': 'adapter',
            '4-bit': 'quantization_4bit',
            'qlora': 'quantization_4bit',
            'quantization': 'quantization_4bit',
            
            // Evaluation
            'loss': 'loss',
            'training loss': 'loss',
            'eval loss': 'eval_loss',
            'validation loss': 'eval_loss',
            'bleu': 'bleu_score',
            'rouge': 'rouge_score',
            
            // Actions
            'start training': 'start_training',
            'stop training': 'stop_training',
            'export': 'export_adapter',
            'merge': 'merge_adapter',
            'model comparison': 'model_comparison',
            'compare models': 'model_comparison',
            'a/b test': 'ab_testing',
            
            // Data & templates
            'data cleaning': 'data_cleaning',
            'alpaca': 'alpaca_format',
            'servicenow': 'template_servicenow',
            'knowledge article': 'template_kb',
            'ticket triage': 'template_triage',
            'runbook': 'template_runbook',
            
            // Post-tuning
            'before/after': 'test_before_after',
            'before and after': 'test_before_after',
            'rating': 'test_rating',
            'effectiveness': 'effectiveness_dashboard',
            'offline': 'offline_mode',
            
            // Navigation / Workflow steps
            'data preparation': 'step_data_prep',
            'model selection': 'step_model_select',
            'training': 'step_training',
            'evaluation': 'step_evaluation',
            'export': 'step_export',
            'model comparison': 'tool_model_compare',
            'hpo (auto-tune)': 'tool_hpo',
            'auto-tune': 'tool_hpo',
            'post-tuning tests': 'tool_post_tuning',
            'post-tuning': 'tool_post_tuning',
        }};
        
        // Track which elements already have listeners
        const processedElements = new WeakSet();
        
        // Set up helpers for Streamlit native elements
        function setupStreamlitHelpers() {{
            // Find labels and add hover handlers
            const selectors = [
                'label',
                '.stMarkdown p',
                '.stMarkdown h3',
                '.stMarkdown h4',
                '.stRadio > label',
                '.stSelectbox label',
                '.stNumberInput label',
                '.stSlider label',
                '.stCheckbox label',
                '[data-testid="stMetricLabel"]',
                '.stExpander summary',
                '.stButton button',
                '.stTabs button[role="tab"]'
            ];
            
            document.querySelectorAll(selectors.join(', ')).forEach(el => {{
                if (processedElements.has(el)) return;
                processedElements.add(el);
                
                const text = el.textContent.toLowerCase();
                
                for (const [pattern, key] of Object.entries(labelMappings)) {{
                    if (text.includes(pattern)) {{
                        el.style.cursor = 'help';
                        el.setAttribute('data-help-key', key);
                        
                        el.addEventListener('mouseenter', function() {{
                            updateHelp(this.getAttribute('data-help-key'));
                        }});
                        el.addEventListener('mouseleave', clearHelp);
                        break;
                    }}
                }}
            }});
            
            // Also handle explicit data-help attributes
            document.querySelectorAll('[data-help]').forEach(el => {{
                if (processedElements.has(el)) return;
                processedElements.add(el);
                
                el.addEventListener('mouseenter', function() {{
                    updateHelp(this.getAttribute('data-help'));
                }});
                el.addEventListener('mouseleave', clearHelp);
            }});
        }}
        
        // Initialize help system
        function initHelpSystem() {{
            if (initTimeout) clearTimeout(initTimeout);
            initTimeout = setTimeout(() => {{
                setupStreamlitHelpers();
            }}, 150);
        }}
        
        // Run on load
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initHelpSystem);
        }} else {{
            initHelpSystem();
        }}
        
        // Re-initialize after Streamlit updates (debounced)
        const observer = new MutationObserver(initHelpSystem);
        observer.observe(document.body, {{ childList: true, subtree: true }});
    }})();
    </script>
    """, unsafe_allow_html=True)


def init_help_system():
    """Initialize the help system. Call this once in your main app."""
    inject_help_system_js()


def help_label(text: str, help_key: str, show_indicator: bool = True) -> str:
    """Create a label with help hover functionality.
    
    Args:
        text: The label text
        help_key: Key to look up in HELP_TEXTS
        show_indicator: Whether to show a small ? icon
    
    Returns:
        HTML string for the label
    """
    indicator = '<span class="help-indicator">?</span>' if show_indicator else ''
    return f'<span class="label-with-help" data-help="{help_key}">{text}{indicator}</span>'


def with_help(label: str, help_key: str) -> str:
    """Wrapper to add help to Streamlit widget labels.
    
    Use with st.markdown before a widget to add help hover.
    
    Example:
        st.markdown(with_help("Learning Rate", "learning_rate"), unsafe_allow_html=True)
        lr = st.number_input("Learning Rate", ...)
    """
    help_info = get_help_text(help_key)
    if help_info:
        return f"""<div class="hoverable-help" data-help="{help_key}" 
                    style="margin-bottom: -0.5rem; padding: 0.25rem 0;">
                    <span style="font-size: 0.875rem; font-weight: 500;">{label}</span>
                    <span class="help-indicator">?</span>
                </div>"""
    return f"**{label}**"
