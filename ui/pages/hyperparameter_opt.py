"""
Hyperparameter Optimization Page - Milestone 5

UI for automated hyperparameter tuning:
- Preset selection
- Custom search space configuration  
- Progress monitoring
- Results visualization
"""

import streamlit as st
from pathlib import Path
import sys
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_hyperparameter_optimization():
    """Render the hyperparameter optimization page."""
    st.title("🔬 Hyperparameter Optimization")
    st.markdown("Automatically find the best training configuration for your model and data.")
    
    # Initialize session state
    if "hpo_running" not in st.session_state:
        st.session_state.hpo_running = False
    if "hpo_results" not in st.session_state:
        st.session_state.hpo_results = None
    if "hpo_progress" not in st.session_state:
        st.session_state.hpo_progress = {"current": 0, "total": 0, "trials": []}
    if "hpo_optimizer" not in st.session_state:
        st.session_state.hpo_optimizer = None
    
    # Check prerequisites
    if not st.session_state.training_samples:
        st.warning("⚠️ Please load training data first (Step 1)")
        return
    
    if not st.session_state.selected_model:
        st.warning("⚠️ Please select a model first (Step 2)")
        return
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Progress", "📋 Results"])
    
    with tab1:
        render_hpo_configuration()
    
    with tab2:
        render_hpo_progress()
    
    with tab3:
        render_hpo_results()


def render_hpo_configuration():
    """Render HPO configuration interface."""
    st.subheader("⚙️ Optimization Configuration")
    
    # Current setup summary
    with st.expander("📋 Current Setup", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Model:**")
            st.code(st.session_state.selected_model[:30] + "..." 
                   if len(st.session_state.selected_model) > 30 
                   else st.session_state.selected_model)
        
        with col2:
            st.markdown("**Training Samples:**")
            st.code(f"{len(st.session_state.training_samples)} samples")
        
        with col3:
            st.markdown("**Hardware:**")
            try:
                import torch
                import psutil
                if torch.cuda.is_available():
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    st.code(f"🎮 GPU: {vram:.1f} GB VRAM")
                else:
                    ram = psutil.virtual_memory()
                    ram_total = ram.total / (1024**3)
                    cpu_count = psutil.cpu_count(logical=True)
                    st.code(f"🖥️ CPU: {cpu_count} cores, {ram_total:.0f} GB RAM")
            except ImportError:
                st.code("Unknown (install psutil)")
            except:
                st.code("Unknown")
    
    st.divider()
    
    # Search method selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        search_method = st.radio(
            "Search Method",
            ["Quick Test", "Random Search", "Smart Search", "Grid Search"],
            help="Choose how to explore the hyperparameter space"
        )
    
    with col2:
        method_descriptions = {
            "Quick Test": "🚀 **Quick Test** - 2 trials to verify setup (~10 min)",
            "Random Search": "🎲 **Random Search** - Sample random configurations",
            "Smart Search": "🧠 **Smart Search** - Focus on promising regions",
            "Grid Search": "📊 **Grid Search** - Exhaustive search (many trials)",
        }
        st.info(method_descriptions.get(search_method, ""))
    
    st.divider()
    
    # Preset selection
    st.subheader("🎯 Optimization Preset")
    
    preset = st.selectbox(
        "Select a preset",
        [
            "Quick Test (2 trials)",
            "Learning Rate Sweep (6 trials)", 
            "LoRA Optimization (12 trials)",
            "Balanced (10 trials)",
            "Full Optimization (20 trials)",
            "Custom Configuration",
        ],
        help="Choose a predefined optimization strategy or customize"
    )
    
    # Load preset configuration
    search_space, n_trials = get_preset_config(preset, search_method)
    
    # Show/edit search space
    show_custom = preset == "Custom Configuration" or st.checkbox("Customize search space")
    
    if show_custom:
        st.markdown("#### Search Space Configuration")
        search_space = render_search_space_editor(search_space)
        
        col1, col2 = st.columns(2)
        with col1:
            n_trials = st.number_input(
                "Number of Trials",
                min_value=1,
                max_value=50,
                value=n_trials,
                help="More trials = better results but longer time"
            )
    
    # Time estimate
    st.divider()
    st.subheader("⏱️ Time Estimate")
    
    estimate = estimate_optimization_time(search_space, search_method, n_trials)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trials", estimate["total_trials"])
    
    with col2:
        st.metric("Est. Time/Trial", f"~{estimate['minutes_per_trial']:.0f} min")
    
    with col3:
        st.metric("Total Est. Time", estimate["formatted"])
    
    # Memory warning
    try:
        import torch
        import psutil
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram < 8 and any(l > 512 for l in search_space.get("max_seq_lengths", [512])):
                st.warning("⚠️ Some configurations may exceed your VRAM. Consider reducing max_seq_length.")
        else:
            # CPU mode warning
            ram = psutil.virtual_memory()
            ram_available = ram.available / (1024**3)
            st.info(f"🖥️ **CPU Mode:** Training will use system RAM ({ram_available:.1f} GB available). Expect ~5-10x longer training times.")
    except:
        pass
    
    # Start button
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.session_state.hpo_running:
            if st.button("⏹️ Stop Optimization", type="secondary", use_container_width=True):
                stop_optimization()
        else:
            if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
                start_optimization(search_space, search_method, n_trials)
                st.rerun()


def get_preset_config(preset: str, search_method: str):
    """Get search space and n_trials from preset."""
    from core.hyperparameter_optimizer import HyperparameterSpace, HPOPresets
    
    if "Quick Test" in preset:
        space, settings = HPOPresets.quick_test()
    elif "Learning Rate" in preset:
        space, settings = HPOPresets.learning_rate_sweep()
    elif "LoRA" in preset:
        space, settings = HPOPresets.lora_optimization()
    elif "Balanced" in preset:
        space, settings = HPOPresets.balanced_optimization()
    elif "Full" in preset:
        space, settings = HPOPresets.full_optimization()
    else:
        space = HyperparameterSpace.balanced_search()
        settings = {"n_trials": 10}
    
    # Convert to dict for editing
    space_dict = {
        "learning_rates": space.learning_rates,
        "lora_r_values": space.lora_r_values,
        "lora_alpha_values": space.lora_alpha_values,
        "lora_dropout_values": space.lora_dropout_values,
        "batch_sizes": space.batch_sizes,
        "gradient_accumulation_steps": space.gradient_accumulation_steps,
        "warmup_ratios": space.warmup_ratios,
        "max_seq_lengths": space.max_seq_lengths,
        "epochs": space.epochs,
    }
    
    return space_dict, settings.get("n_trials", 10)


def render_search_space_editor(space: Dict) -> Dict:
    """Render editor for search space configuration."""
    edited_space = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Learning & Training**")
        
        lr_options = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]
        edited_space["learning_rates"] = st.multiselect(
            "Learning Rates",
            lr_options,
            default=[lr for lr in space.get("learning_rates", [1e-4]) if lr in lr_options],
            format_func=lambda x: f"{x:.0e}"
        )
        
        edited_space["epochs"] = st.multiselect(
            "Epochs",
            [1, 2, 3, 5, 10],
            default=space.get("epochs", [3])
        )
        
        edited_space["batch_sizes"] = st.multiselect(
            "Batch Sizes",
            [1, 2, 4],
            default=space.get("batch_sizes", [1])
        )
        
        edited_space["gradient_accumulation_steps"] = st.multiselect(
            "Gradient Accumulation",
            [4, 8, 16, 32],
            default=space.get("gradient_accumulation_steps", [16])
        )
    
    with col2:
        st.markdown("**LoRA Configuration**")
        
        edited_space["lora_r_values"] = st.multiselect(
            "LoRA Rank (r)",
            [4, 8, 16, 32, 64],
            default=space.get("lora_r_values", [16])
        )
        
        edited_space["lora_alpha_values"] = st.multiselect(
            "LoRA Alpha",
            [8, 16, 32, 64, 128],
            default=space.get("lora_alpha_values", [32])
        )
        
        edited_space["lora_dropout_values"] = st.multiselect(
            "LoRA Dropout",
            [0.0, 0.05, 0.1, 0.2],
            default=space.get("lora_dropout_values", [0.05])
        )
        
        edited_space["max_seq_lengths"] = st.multiselect(
            "Max Sequence Length",
            [128, 256, 512, 1024],
            default=space.get("max_seq_lengths", [512])
        )
        
        edited_space["warmup_ratios"] = st.multiselect(
            "Warmup Ratio",
            [0.0, 0.03, 0.1],
            default=space.get("warmup_ratios", [0.03])
        )
    
    # Calculate total combinations
    total = 1
    for key, values in edited_space.items():
        total *= len(values) if values else 1
    
    st.caption(f"Total possible combinations: {total:,}")
    
    return edited_space


def estimate_optimization_time(search_space: Dict, search_method: str, n_trials: int) -> Dict:
    """Estimate optimization time."""
    # Estimate minutes per trial based on data size and seq length
    samples = len(st.session_state.training_samples) if st.session_state.training_samples else 100
    max_seq = max(search_space.get("max_seq_lengths", [512]))
    
    # Base estimate: ~2 min for quick, scale with samples and seq length
    base_minutes = 2.0
    sample_factor = min(samples / 100, 5)  # Cap at 5x
    seq_factor = max_seq / 256
    
    minutes_per_trial = base_minutes * sample_factor * seq_factor
    minutes_per_trial = max(2, min(minutes_per_trial, 30))  # Clamp between 2-30 min
    
    total_minutes = n_trials * minutes_per_trial
    
    if total_minutes < 60:
        formatted = f"~{total_minutes:.0f} minutes"
    else:
        formatted = f"~{total_minutes/60:.1f} hours"
    
    return {
        "total_trials": n_trials,
        "minutes_per_trial": minutes_per_trial,
        "total_minutes": total_minutes,
        "formatted": formatted,
    }


def start_optimization(search_space: Dict, search_method: str, n_trials: int):
    """Start the optimization process."""
    from core.hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterSpace
    from core.dataset_handler import DatasetHandler
    
    st.session_state.hpo_running = True
    st.session_state.hpo_progress = {"current": 0, "total": n_trials, "trials": []}
    
    # Create search space object
    space = HyperparameterSpace(
        learning_rates=search_space.get("learning_rates", [1e-4]),
        lora_r_values=search_space.get("lora_r_values", [16]),
        lora_alpha_values=search_space.get("lora_alpha_values", [32]),
        lora_dropout_values=search_space.get("lora_dropout_values", [0.05]),
        batch_sizes=search_space.get("batch_sizes", [1]),
        gradient_accumulation_steps=search_space.get("gradient_accumulation_steps", [16]),
        warmup_ratios=search_space.get("warmup_ratios", [0.03]),
        max_seq_lengths=search_space.get("max_seq_lengths", [512]),
        epochs=search_space.get("epochs", [3]),
    )
    
    # Prepare dataset
    handler = DatasetHandler()
    train_ds = handler.to_hf_dataset(
        st.session_state.training_samples,
        prompt_template="alpaca",
    )
    
    # Create optimizer
    output_dir = f"./models/hpo_runs/hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    optimizer = HyperparameterOptimizer(
        model_name=st.session_state.selected_model,
        train_dataset=train_ds,
        output_dir=output_dir,
    )
    
    st.session_state.hpo_optimizer = optimizer
    
    # Run optimization in background thread
    def run_hpo():
        try:
            # Set progress callback
            def progress_callback(current, total, trial):
                st.session_state.hpo_progress["current"] = current
                st.session_state.hpo_progress["total"] = total
                st.session_state.hpo_progress["trials"].append(trial.to_dict())
            
            optimizer.set_progress_callback(progress_callback)
            
            # Run appropriate search method
            method_map = {
                "Quick Test": "random",
                "Random Search": "random", 
                "Smart Search": "smart",
                "Grid Search": "grid",
            }
            method = method_map.get(search_method, "random")
            
            if method == "grid":
                results = optimizer.grid_search(space, max_trials=n_trials)
            elif method == "smart":
                results = optimizer.smart_search(space, n_trials=n_trials)
            else:
                results = optimizer.random_search(space, n_trials=n_trials)
            
            st.session_state.hpo_results = results.to_dict()
            
        except Exception as e:
            st.session_state.hpo_error = str(e)
        finally:
            st.session_state.hpo_running = False
    
    thread = threading.Thread(target=run_hpo)
    thread.start()


def stop_optimization():
    """Stop the running optimization."""
    if st.session_state.hpo_optimizer:
        st.session_state.hpo_optimizer.stop()
    st.session_state.hpo_running = False


def render_hpo_progress():
    """Render optimization progress."""
    st.subheader("📊 Optimization Progress")
    
    if not st.session_state.hpo_running and not st.session_state.hpo_progress.get("trials"):
        st.info("No optimization running. Configure and start an optimization in the Configuration tab.")
        return
    
    progress = st.session_state.hpo_progress
    
    # Progress bar
    current = progress.get("current", 0)
    total = progress.get("total", 1)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(current / total if total > 0 else 0)
    
    with col2:
        st.markdown(f"**{current} / {total}** trials")
    
    if st.session_state.hpo_running:
        st.info("🔄 Optimization in progress... This page updates automatically.")
        
        # Auto-refresh
        time.sleep(2)
        st.rerun()
    
    # Show trial history
    trials = progress.get("trials", [])
    
    if trials:
        st.divider()
        st.markdown("#### Trial History")
        
        # Convert to dataframe
        trial_data = []
        for t in trials:
            trial_data.append({
                "Trial": t["trial_id"],
                "Loss": f"{t['final_loss']:.4f}" if t['final_loss'] < float('inf') else "N/A",
                "LR": f"{t['hyperparameters'].get('learning_rate', 0):.0e}",
                "LoRA r": t['hyperparameters'].get('lora_r', 'N/A'),
                "Alpha": t['hyperparameters'].get('lora_alpha', 'N/A'),
                "Status": "✅" if t['status'] == 'completed' else "❌" if t['status'] == 'failed' else "⏳",
                "Time": f"{t['training_time_seconds']/60:.1f}m" if t['training_time_seconds'] else "N/A",
            })
        
        df = pd.DataFrame(trial_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Loss chart
        if len(trials) > 1:
            st.markdown("#### Loss Progress")
            
            completed_trials = [t for t in trials if t['status'] == 'completed']
            if completed_trials:
                chart_data = pd.DataFrame([
                    {"Trial": t["trial_id"], "Loss": t["final_loss"]}
                    for t in completed_trials
                ])
                st.line_chart(chart_data.set_index("Trial"))


def render_hpo_results():
    """Render optimization results."""
    st.subheader("📋 Optimization Results")
    
    # Check for previous results
    results = st.session_state.hpo_results
    
    # Also check for saved results
    hpo_dir = Path("./models/hpo_runs")
    saved_results = []
    
    if hpo_dir.exists():
        for run_dir in hpo_dir.iterdir():
            if run_dir.is_dir():
                results_file = run_dir / "optimization_results.json"
                if results_file.exists():
                    try:
                        with open(results_file) as f:
                            saved_results.append({
                                "name": run_dir.name,
                                "path": str(results_file),
                                "data": json.load(f)
                            })
                    except:
                        pass
    
    if not results and not saved_results:
        st.info("No optimization results yet. Run an optimization first.")
        return
    
    # Result selector
    if saved_results:
        st.markdown("#### Available Results")
        
        result_options = ["Current Session"] if results else []
        result_options.extend([r["name"] for r in saved_results])
        
        selected = st.selectbox("Select results to view", result_options)
        
        if selected == "Current Session":
            display_results = results
        else:
            display_results = next(r["data"] for r in saved_results if r["name"] == selected)
    else:
        display_results = results
    
    if not display_results:
        return
    
    # Summary metrics
    st.divider()
    st.markdown("#### 🏆 Best Configuration Found")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Best Loss",
            f"{display_results['best_loss']:.4f}" if display_results['best_loss'] < float('inf') else "N/A"
        )
    
    with col2:
        st.metric("Best Trial", f"#{display_results['best_trial_id']}")
    
    with col3:
        st.metric(
            "Completed",
            f"{display_results['completed_trials']}/{display_results['total_trials']}"
        )
    
    # Best hyperparameters
    st.markdown("#### Best Hyperparameters")
    
    best_params = display_results.get("best_hyperparameters", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training:**")
        st.code(f"""
learning_rate: {best_params.get('learning_rate', 'N/A')}
epochs: {best_params.get('epochs', 'N/A')}
batch_size: {best_params.get('batch_size', 'N/A')}
gradient_accumulation: {best_params.get('gradient_accumulation_steps', 'N/A')}
warmup_ratio: {best_params.get('warmup_ratio', 'N/A')}
max_seq_length: {best_params.get('max_seq_length', 'N/A')}
        """)
    
    with col2:
        st.markdown("**LoRA:**")
        st.code(f"""
lora_r: {best_params.get('lora_r', 'N/A')}
lora_alpha: {best_params.get('lora_alpha', 'N/A')}
lora_dropout: {best_params.get('lora_dropout', 'N/A')}
        """)
    
    # All trials table
    st.divider()
    st.markdown("#### All Trials")
    
    trials = display_results.get("trials", [])
    if trials:
        # Sort by loss
        sorted_trials = sorted(
            [t for t in trials if t['status'] == 'completed'],
            key=lambda x: x['final_loss']
        )
        
        trial_data = []
        for i, t in enumerate(sorted_trials):
            trial_data.append({
                "Rank": i + 1,
                "Trial": t["trial_id"],
                "Loss": f"{t['final_loss']:.4f}",
                "LR": f"{t['hyperparameters'].get('learning_rate', 0):.0e}",
                "r": t['hyperparameters'].get('lora_r', 'N/A'),
                "α": t['hyperparameters'].get('lora_alpha', 'N/A'),
                "Epochs": t['hyperparameters'].get('epochs', 'N/A'),
                "Seq Len": t['hyperparameters'].get('max_seq_length', 'N/A'),
                "Time": f"{t['training_time_seconds']/60:.1f}m" if t.get('training_time_seconds') else "N/A",
            })
        
        df = pd.DataFrame(trial_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Actions
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Results", use_container_width=True):
            st.download_button(
                "Download JSON",
                data=json.dumps(display_results, indent=2),
                file_name=f"hpo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
    
    with col2:
        if st.button("🚀 Apply Best Config", use_container_width=True):
            apply_best_config(best_params)
            st.success("Best configuration applied! Go to Training tab to use it.")
    
    with col3:
        if st.button("📊 Compare in Dashboard", use_container_width=True):
            st.info("Navigate to Model Comparison to compare trained models.")


def apply_best_config(params: Dict):
    """Apply best configuration to session state for training."""
    st.session_state.hpo_best_config = params
    st.session_state.training_config = {
        "learning_rate": params.get("learning_rate", 2e-4),
        "epochs": params.get("epochs", 3),
        "batch_size": params.get("batch_size", 1),
        "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 16),
        "lora_r": params.get("lora_r", 16),
        "lora_alpha": params.get("lora_alpha", 32),
        "lora_dropout": params.get("lora_dropout", 0.05),
        "max_seq_length": params.get("max_seq_length", 512),
        "warmup_ratio": params.get("warmup_ratio", 0.03),
    }


if __name__ == "__main__":
    render_hyperparameter_optimization()
