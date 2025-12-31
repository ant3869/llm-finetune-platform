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


def render_training():
    """Render the training page."""
    st.title("🚀 Step 3: Training")
    st.markdown("Configure training parameters and start fine-tuning.")
    
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data**")
        samples = st.session_state.training_samples
        stats = st.session_state.dataset_stats
        st.markdown(f"""
        - Samples: {len(samples)}
        - Training: {stats.train_samples}
        - Validation: {stats.validation_samples}
        """)
    
    with col2:
        st.markdown("**Model**")
        model_config = st.session_state.model_config
        st.markdown(f"""
        - {model_config['name']}
        - Size: {model_config['size']}
        - VRAM: {model_config['vram']}
        """)
    
    with col3:
        st.markdown("**Hardware**")
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                st.markdown(f"""
                - GPU: {gpu[:20]}...
                - VRAM: {vram:.1f} GB
                - Status: ✅ Ready
                """)
            else:
                st.markdown("""
                - GPU: Not available
                - Status: ⚠️ CPU only
                """)
        except:
            st.markdown("- Status: Unknown")


def render_training_config():
    """Render training configuration options."""
    st.subheader("⚙️ Training Configuration")
    
    # Preset selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        preset = st.selectbox(
            "Training Preset",
            ["quick_test", "balanced", "thorough", "custom"],
            index=1,
            help="Choose a preset or customize settings"
        )
    
    with col2:
        preset_descriptions = {
            "quick_test": "⚡ Fast test run (1 epoch, 50 steps) - Verify setup works",
            "balanced": "⚖️ Balanced training (3 epochs) - Good quality, reasonable time",
            "thorough": "🎯 Thorough training (5 epochs) - Best quality, longer training",
            "custom": "✏️ Custom settings - Full control over all parameters",
        }
        st.info(preset_descriptions.get(preset, ""))
    
    # Load preset config
    try:
        if preset != "custom":
            config = TrainingConfig.from_preset(preset)
        else:
            config = TrainingConfig()
    except:
        config = TrainingConfig()
    
    # Advanced settings
    show_advanced = preset == "custom" or st.checkbox("Show advanced settings")
    
    if show_advanced:
        st.markdown("#### Training Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=20,
                value=config.epochs,
                help="Number of training passes through the data"
            )
        
        with col2:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=8,
                value=config.batch_size,
                help="Samples per batch (keep low for 8GB VRAM)"
            )
        
        with col3:
            grad_accum = st.number_input(
                "Gradient Accumulation",
                min_value=1,
                max_value=64,
                value=config.gradient_accumulation_steps,
                help="Accumulate gradients over N steps"
            )
        
        with col4:
            effective_batch = batch_size * grad_accum
            st.metric("Effective Batch", effective_batch)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=config.learning_rate,
                format="%.2e",
                help="Learning rate for optimizer"
            )
        
        with col2:
            max_seq_length = st.selectbox(
                "Max Sequence Length",
                [256, 512, 1024, 2048],
                index=[256, 512, 1024, 2048].index(config.max_seq_length) if config.max_seq_length in [256, 512, 1024, 2048] else 1,
                help="Maximum tokens per sample"
            )
        
        with col3:
            lora_r = st.selectbox(
                "LoRA Rank (r)",
                [8, 16, 32, 64],
                index=[8, 16, 32, 64].index(config.lora_r) if config.lora_r in [8, 16, 32, 64] else 1,
                help="LoRA rank - higher = more parameters"
            )
        
        with col4:
            lora_alpha = st.selectbox(
                "LoRA Alpha",
                [16, 32, 64, 128],
                index=[16, 32, 64, 128].index(config.lora_alpha) if config.lora_alpha in [16, 32, 64, 128] else 1,
                help="LoRA alpha - usually 2x rank"
            )
        
        # Memory optimization
        st.markdown("#### Memory Optimization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gradient_checkpointing = st.checkbox(
                "Gradient Checkpointing",
                value=config.gradient_checkpointing,
                help="Reduces VRAM at cost of speed"
            )
        
        with col2:
            fp16 = st.checkbox(
                "FP16 Training",
                value=config.fp16,
                help="Use mixed precision training"
            )
        
        with col3:
            max_steps = st.number_input(
                "Max Steps (0 = use epochs)",
                min_value=0,
                max_value=10000,
                value=max(0, config.max_steps),
                help="Limit total steps (0 to use epochs)"
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
    st.markdown("#### 💾 Estimated VRAM Usage")
    
    from core.model_loader import ModelLoader
    loader = ModelLoader()
    estimate = loader.estimate_training_vram(
        model_size_gb=4.0,  # Approximate
        batch_size=batch_size,
        seq_length=max_seq_length
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", f"{estimate['model']:.1f} GB")
    with col2:
        st.metric("LoRA", f"{estimate['lora_adapters']:.1f} GB")
    with col3:
        st.metric("Optimizer", f"{estimate['optimizer']:.1f} GB")
    with col4:
        color = "normal" if estimate['fits_8gb'] else "inverse"
        st.metric("Total", f"{estimate['total']:.1f} GB", delta="✓ OK" if estimate['fits_8gb'] else "⚠️ High")


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
        st.markdown("### Training Progress")
        
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
                st.info(f"Adapter saved to: `{st.session_state.trained_adapter_path}`")
            
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
        # Start training button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("---")
            
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
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
