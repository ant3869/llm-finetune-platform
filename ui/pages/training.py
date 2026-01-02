"""
Training Page - Step 3

Configure and run fine-tuning with real-time progress monitoring.
"""

import streamlit as st
from pathlib import Path
import sys
import threading
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.trainer import Trainer, TrainingConfig, TrainingProgress
from core.dataset_handler import DatasetHandler
from ui.components.progress_tracker import ProgressTracker, render_training_status


def render_info_box(title: str, content: str, icon: str = "ℹ️"):
    """Render a styled info box with explanation."""
    st.markdown(f"""
    <div style="background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6; 
                padding: 0.75rem 1rem; border-radius: 0 0.5rem 0.5rem 0; margin: 0.5rem 0;">
        <div style="font-weight: 600; color: #3b82f6; margin-bottom: 0.25rem;">{icon} {title}</div>
        <div style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.5;">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_training():
    """Render the training page."""
    st.title("🚀 Step 3: Training")
    
    # Main page explanation
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); 
                padding: 1rem 1.25rem; border-radius: 0.5rem; margin-bottom: 1.5rem; border: 1px solid rgba(59, 130, 246, 0.2);">
        <p style="margin: 0; color: var(--text-secondary); line-height: 1.6;">
            <strong>What happens here:</strong> The model learns from your training data by adjusting its parameters. 
            You'll choose training settings (or use a preset), then click <strong>Start Training</strong>. 
            The process typically takes 10-60 minutes depending on your data size and settings.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check prerequisites
    if not st.session_state.training_samples:
        st.warning("⚠️ Please load training data first (Step 1)")
        if st.button("← Back to Data Preparation"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    if not st.session_state.selected_model:
        st.warning("⚠️ Please select a model first (Step 2)")
        if st.button("← Back to Model Selection"):
            st.session_state.current_step = 2
            st.rerun()
        return
    
    # Initialize progress tracker in session state
    if "progress_tracker" not in st.session_state:
        st.session_state.progress_tracker = ProgressTracker()
    
    # Show current setup summary
    render_setup_summary()
    
    st.divider()
    
    # Training configuration
    if not st.session_state.is_training:
        render_training_config()
    
    st.divider()
    
    # Training controls and progress
    render_training_controls()


def render_setup_summary():
    """Render summary of current setup."""
    st.subheader("📋 Training Setup")
    st.caption("Review your data, model, and hardware before training")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📄 Data**")
        samples = st.session_state.training_samples
        stats = st.session_state.dataset_stats
        st.markdown(f"""
        - **Samples:** {len(samples)}
        - **Training:** {stats.train_samples} (90%)
        - **Validation:** {stats.validation_samples} (10%)
        """)
        with st.expander("What's this?", expanded=False):
            st.markdown("""
            Your data is split into **training** (used to teach the model) and 
            **validation** (used to check if learning is working). 
            More samples = better learning, but quality matters most.
            """)
    
    with col2:
        st.markdown("**🤖 Model**")
        model_config = st.session_state.model_config
        st.markdown(f"""
        - **{model_config['name']}**
        - **Size:** {model_config['size']} parameters
        - **VRAM:** {model_config['vram']} needed
        """)
        with st.expander("What's this?", expanded=False):
            st.markdown("""
            The **base model** you're fine-tuning. Larger models (7B+) are smarter 
            but need more GPU memory. We use **4-bit quantization** to fit them in 8GB.
            """)
    
    with col3:
        st.markdown("**💻 Hardware**")
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                st.markdown(f"""
                - **GPU:** {gpu[:20]}...
                - **VRAM:** {vram:.1f} GB
                - **Status:** ✅ Ready
                """)
            else:
                st.markdown("""
                - **GPU:** Not available
                - **Status:** ⚠️ CPU only (very slow)
                """)
        except:
            st.markdown("- **Status:** Unknown")
        with st.expander("What's this?", expanded=False):
            st.markdown("""
            Your **GPU** (graphics card) does the heavy lifting. 
            **VRAM** is GPU memory - you need enough to fit the model and training data.
            """)


def render_training_config():
    """Render training configuration options."""
    st.subheader("⚙️ Training Configuration")
    st.caption("Choose how the model should learn from your data")
    
    # Preset selection with better explanation
    render_info_box(
        "Choose a Training Preset",
        "Presets are pre-configured settings for different goals. Start with <b>balanced</b> for most cases. "
        "Use <b>quick_test</b> first to verify everything works before a full training run.",
        "🎯"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        preset = st.selectbox(
            "Training Preset",
            ["quick_test", "balanced", "thorough", "custom"],
            index=1,
            help="Choose a preset or customize settings",
            format_func=lambda x: {
                "quick_test": "⚡ Quick Test",
                "balanced": "⚖️ Balanced (Recommended)",
                "thorough": "🎯 Thorough",
                "custom": "✏️ Custom"
            }.get(x, x)
        )
    
    with col2:
        preset_info = {
            "quick_test": ("⚡ Quick Test", "~5 minutes", "1 epoch, 50 steps max", 
                          "Use this first to verify your setup works. Not for real training."),
            "balanced": ("⚖️ Balanced", "~30-60 minutes", "3 epochs, 512 seq length",
                        "Good quality results in reasonable time. Best for most users."),
            "thorough": ("🎯 Thorough", "~1-2 hours", "5 epochs, 1024 seq length",
                        "Best quality but takes longer. Use when you have time."),
            "custom": ("✏️ Custom", "Varies", "You decide all settings",
                      "Full control over training parameters."),
        }
        name, time_est, settings, desc = preset_info.get(preset, ("", "", "", ""))
        
        st.markdown(f"""
        <div style="background: var(--bg-tertiary); padding: 1rem; border-radius: 0.5rem; height: 100%;">
            <div style="font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;">{name}</div>
            <div style="display: flex; gap: 1rem; margin-bottom: 0.5rem;">
                <span style="background: rgba(59, 130, 246, 0.2); color: #3b82f6; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem;">⏱️ {time_est}</span>
                <span style="background: rgba(139, 92, 246, 0.2); color: #8b5cf6; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem;">📊 {settings}</span>
            </div>
            <div style="font-size: 0.875rem; color: var(--text-secondary);">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load preset config
    try:
        if preset != "custom":
            config = TrainingConfig.from_preset(preset)
        else:
            config = TrainingConfig()
    except:
        config = TrainingConfig()
    
    # Advanced settings
    show_advanced = preset == "custom" or st.checkbox("📐 Show advanced settings", 
        help="View and modify individual training parameters")
    
    if show_advanced:
        st.markdown("---")
        st.markdown("#### 📊 Training Parameters")
        st.caption("Fine-tune how the model learns. Hover over (?) icons for explanations.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=20,
                value=config.epochs,
                help="How many times to go through all your data. More = better learning but longer training. Usually 3-5 is good."
            )
            st.caption("Passes through data")
        
        with col2:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=8,
                value=config.batch_size,
                help="Samples processed together. Keep at 1-2 for 8GB GPU. Higher = faster but uses more memory."
            )
            st.caption("Samples per batch")
        
        with col3:
            grad_accum = st.number_input(
                "Gradient Accumulation",
                min_value=1,
                max_value=64,
                value=config.gradient_accumulation_steps,
                help="Simulates larger batches without more memory. 8-16 is typical."
            )
            st.caption("Steps before update")
        
        with col4:
            effective_batch = batch_size * grad_accum
            st.metric("Effective Batch", effective_batch, help="Total samples before each model update")
            st.caption("batch × accumulation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=config.learning_rate,
                format="%.2e",
                help="How fast the model learns. Too high = unstable, too low = slow. 1e-4 to 2e-4 is usually good."
            )
            st.caption("Learning speed")
        
        with col2:
            max_seq_length = st.selectbox(
                "Max Sequence Length",
                [256, 512, 1024, 2048],
                index=[256, 512, 1024, 2048].index(config.max_seq_length) if config.max_seq_length in [256, 512, 1024, 2048] else 1,
                help="Max tokens per sample. Longer = more context but more memory. Match to your typical input+output length."
            )
            st.caption("Tokens per sample")
        
        with col3:
            lora_r = st.selectbox(
                "LoRA Rank (r)",
                [8, 16, 32, 64],
                index=[8, 16, 32, 64].index(config.lora_r) if config.lora_r in [8, 16, 32, 64] else 1,
                help="Size of adaptation. Higher = more learning capacity but more memory. 16 is a good default."
            )
            st.caption("Adaptation size")
        
        with col4:
            lora_alpha = st.selectbox(
                "LoRA Alpha",
                [16, 32, 64, 128],
                index=[16, 32, 64, 128].index(config.lora_alpha) if config.lora_alpha in [16, 32, 64, 128] else 1,
                help="Scaling factor. Usually 2× the rank. Higher = stronger adaptation effect."
            )
            st.caption("Usually 2× rank")
        
        # Memory optimization
        st.markdown("#### 💾 Memory Optimization")
        st.caption("Settings to reduce GPU memory usage")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gradient_checkpointing = st.checkbox(
                "Gradient Checkpointing",
                value=config.gradient_checkpointing,
                help="Saves memory by recomputing during backward pass. ~30% less VRAM, ~20% slower."
            )
        
        with col2:
            fp16 = st.checkbox(
                "FP16 Training",
                value=config.fp16,
                help="Use 16-bit precision. Faster and uses less memory with minimal quality impact."
            )
        
        with col3:
            max_steps = st.number_input(
                "Max Steps (0 = use epochs)",
                min_value=0,
                max_value=10000,
                value=max(0, config.max_steps),
                help="Limit total training steps. 0 means train for full epochs instead."
            )
        
        # Update config
        config.epochs = epochs
        config.batch_size = batch_size
        config.gradient_accumulation_steps = grad_accum
        config.learning_rate = learning_rate
        config.max_seq_length = max_seq_length
        config.lora_r = lora_r
        config.lora_alpha = lora_alpha
        config.gradient_checkpointing = gradient_checkpointing
        config.fp16 = fp16
        config.max_steps = max_steps if max_steps > 0 else -1
    
    else:
        # Use preset defaults
        epochs = config.epochs
        batch_size = config.batch_size
        grad_accum = config.gradient_accumulation_steps
        learning_rate = config.learning_rate
        max_seq_length = config.max_seq_length
        lora_r = config.lora_r
        lora_alpha = config.lora_alpha
    
    # Store config
    config.model_name_or_path = st.session_state.selected_model
    st.session_state.training_config = config
    
    # VRAM estimate
    st.markdown("---")
    st.markdown("#### 💾 Estimated VRAM Usage")
    st.caption("Will training fit in your GPU memory?")
    
    from core.model_loader import ModelLoader
    loader = ModelLoader()
    estimate = loader.estimate_training_vram(
        model_size_gb=4.0,  # Approximate
        batch_size=batch_size,
        seq_length=max_seq_length
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", f"{estimate['model']:.1f} GB", help="Base model size (4-bit quantized)")
    with col2:
        st.metric("LoRA", f"{estimate['lora_adapters']:.1f} GB", help="LoRA adapter parameters")
    with col3:
        st.metric("Optimizer", f"{estimate['optimizer']:.1f} GB", help="Optimizer state (AdamW)")
    with col4:
        if estimate['fits_8gb']:
            st.metric("Total", f"{estimate['total']:.1f} GB", delta="✓ OK", delta_color="normal",
                     help="Estimated total VRAM usage - should fit!")
        else:
            st.metric("Total", f"{estimate['total']:.1f} GB", delta="⚠️ High", delta_color="inverse",
                     help="May exceed GPU memory. Try reducing sequence length or batch size.")
    
    if not estimate['fits_8gb']:
        st.warning("⚠️ Estimated VRAM may exceed 8GB. Consider reducing **Max Sequence Length** or **Batch Size**.")


def render_training_controls():
    """Render training controls and progress display."""
    st.subheader("🎮 Training Controls")
    
    tracker = st.session_state.progress_tracker
    
    # Show any stored error from previous run
    if "training_error" in st.session_state and st.session_state.training_error:
        st.error(f"❌ Last training failed: {st.session_state.training_error['message']}")
        with st.expander("🔍 Error Details", expanded=True):
            st.code(st.session_state.training_error['traceback'])
        if st.button("🗑️ Clear Error"):
            st.session_state.training_error = None
            st.rerun()
        st.divider()
    
    if st.session_state.is_training:
        # Show progress
        st.markdown("### 📈 Training Progress")
        
        render_info_box(
            "Training in Progress",
            "The model is learning from your data. This may take 10-60+ minutes depending on settings. "
            "You can stop early if needed - the partial adapter will be saved.",
            "⏳"
        )
        
        # Status and metrics
        tracker.render()
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col2:
            if st.button("🛑 Stop Training", type="secondary", use_container_width=True):
                if st.session_state.trainer:
                    st.session_state.trainer.stop_training()
                    st.session_state.is_training = False
                    st.warning("Training stopped by user")
        
        # Auto-refresh
        if tracker.status == "training":
            time.sleep(1)
            st.rerun()
        
        # Training complete
        if tracker.status == "completed":
            st.session_state.is_training = False
            st.balloons()
            st.success("🎉 Training complete!")
            
            if st.session_state.trained_adapter_path:
                st.info(f"📁 Adapter saved to: `{st.session_state.trained_adapter_path}`")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Train Again"):
                    tracker.reset()
                    st.rerun()
            with col2:
                if st.button("📊 Go to Evaluation →", type="primary"):
                    st.session_state.current_step = 4
                    st.rerun()
    
    else:
        # Pre-training checklist and start button
        st.markdown("#### ✅ Pre-Training Checklist")
        
        checks = [
            ("📄 Training data loaded", bool(st.session_state.training_samples), "Load data in Step 1"),
            ("🤖 Model selected", bool(st.session_state.selected_model), "Select model in Step 2"),
            ("⚙️ Configuration set", bool(st.session_state.training_config), "Choose preset above"),
        ]
        
        all_ready = True
        for label, status, hint in checks:
            if status:
                st.markdown(f"✅ {label}")
            else:
                st.markdown(f"❌ {label} - *{hint}*")
                all_ready = False
        
        st.markdown("---")
        
        # Start training button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            render_info_box(
                "Ready to Train",
                "Click the button below to start fine-tuning. The model will be downloaded (if needed), "
                "loaded into GPU memory, and training will begin. <b>This cannot be undone</b> - but you can stop early.",
                "🚀"
            )
            
            if st.button("🚀 Start Training", type="primary", use_container_width=True, disabled=not all_ready):
                start_training()
        
        # Navigation
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Back to Model Selection", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()


def start_training():
    """Start the training process."""
    config = st.session_state.training_config
    samples = st.session_state.training_samples
    tracker = st.session_state.progress_tracker
    
    # Reset tracker
    tracker.reset()
    tracker.status = "loading"
    
    st.session_state.is_training = True
    
    # This would normally run in a background thread
    # For Streamlit, we'll show progress updates
    
    try:
        with st.spinner("Preparing datasets..."):
            handler = DatasetHandler()
            split_data = handler.prepare_dataset(samples)
            train_dataset = handler.to_hf_dataset(split_data["train"])
            eval_dataset = handler.to_hf_dataset(split_data["validation"]) if split_data["validation"] else None
        
        st.info(f"Training: {len(split_data['train'])} samples | Validation: {len(split_data['validation'])} samples")
        
        with st.spinner("Loading model (this may take a few minutes)..."):
            trainer = Trainer(config)
            trainer.add_progress_callback(tracker.update)
            st.session_state.trainer = trainer
            
            trainer.prepare_model(config.model_name_or_path)
        
        st.success("Model loaded! Starting training...")
        tracker.status = "training"
        
        # Run training (this blocks but callbacks update progress)
        adapter_path = trainer.train(
            train_dataset,
            eval_dataset,
            output_dir="./models/adapters"
        )
        
        st.session_state.trained_adapter_path = str(adapter_path)
        tracker.status = "completed"
        
    except Exception as e:
        tracker.status = "error"
        st.session_state.is_training = False
        
        import traceback
        error_traceback = traceback.format_exc()
        
        # Store error in session state so it persists across reruns
        st.session_state.training_error = {
            "message": str(e),
            "traceback": error_traceback
        }
        
        # Log to console for debugging
        print(f"Training Error: {e}")
        print(error_traceback)
    
    finally:
        if st.session_state.trainer:
            st.session_state.trainer.cleanup()
    
    st.rerun()
