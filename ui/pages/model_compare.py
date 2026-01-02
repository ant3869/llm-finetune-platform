"""
Model Comparison Dashboard - Milestone 5

Compare multiple fine-tuned models side-by-side:
- Performance metrics comparison
- Response quality comparison
- Training metrics visualization
- A/B testing capabilities
"""

import streamlit as st
from pathlib import Path
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_model_compare():
    """Render the model comparison dashboard."""
    st.title("📊 Model Comparison Dashboard")
    st.markdown("Compare multiple fine-tuned models side-by-side to find the best performer.")
    
    # Initialize session state
    if "compare_models" not in st.session_state:
        st.session_state.compare_models = []
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None
    if "compare_engines" not in st.session_state:
        st.session_state.compare_engines = {}
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Setup", 
        "📈 Training Metrics", 
        "🧪 A/B Testing",
        "📋 Results Summary"
    ])
    
    with tab1:
        render_model_selection()
    
    with tab2:
        render_training_metrics_comparison()
    
    with tab3:
        render_ab_testing()
    
    with tab4:
        render_comparison_summary()


def render_model_selection():
    """Render model selection interface."""
    st.subheader("🔧 Select Models to Compare")
    
    # Find available adapters
    adapter_dir = Path("./models/adapters")
    available_adapters = []
    
    if adapter_dir.exists():
        for d in adapter_dir.iterdir():
            if d.is_dir() and d.name != ".gitkeep":
                adapter_info = get_adapter_info(d)
                if adapter_info:
                    available_adapters.append(adapter_info)
    
    if not available_adapters:
        st.info("No trained adapters found. Train some models first to compare them.")
        st.markdown("""
        **To get started:**
        1. Go to **Step 1: Data Preparation** and load training data
        2. Go to **Step 2: Model Selection** and choose a base model
        3. Go to **Step 3: Training** and train with different configurations
        4. Return here to compare your models
        """)
        return
    
    # Display available adapters
    st.markdown(f"**Found {len(available_adapters)} trained adapters:**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Multi-select for adapters
        adapter_names = [a["name"] for a in available_adapters]
        selected_names = st.multiselect(
            "Select adapters to compare (2-4 recommended)",
            adapter_names,
            default=st.session_state.compare_models[:4] if st.session_state.compare_models else adapter_names[:min(2, len(adapter_names))],
            help="Select 2-4 models for meaningful comparison"
        )
        st.session_state.compare_models = selected_names
    
    with col2:
        st.markdown("**Quick Actions:**")
        if st.button("📥 Load Training Configs", disabled=len(selected_names) < 1):
            load_training_configs(available_adapters, selected_names)
    
    # Show selected adapter details
    if selected_names:
        st.divider()
        st.subheader("📋 Selected Models Overview")
        
        # Create comparison table
        comparison_data = []
        for name in selected_names:
            adapter = next((a for a in available_adapters if a["name"] == name), None)
            if adapter:
                comparison_data.append({
                    "Model": name,
                    "Base Model": adapter.get("base_model", "Unknown"),
                    "LoRA Rank": adapter.get("lora_r", "N/A"),
                    "Trained": adapter.get("timestamp", "Unknown"),
                    "Final Loss": f"{adapter.get('final_loss', 'N/A'):.4f}" if isinstance(adapter.get("final_loss"), (int, float)) else "N/A",
                    "Training Time": format_time(adapter.get("training_time", 0)),
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)


def get_adapter_info(adapter_path: Path) -> Optional[Dict[str, Any]]:
    """Extract information about an adapter from its config files."""
    info = {"name": adapter_path.name, "path": str(adapter_path)}
    
    # Try to load training config
    config_file = adapter_path / "training_config.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            info["base_model"] = config.get("model", "Unknown")
            info["lora_r"] = config.get("lora_r", "N/A")
            info["lora_alpha"] = config.get("lora_alpha", "N/A")
            info["epochs"] = config.get("epochs", "N/A")
            info["batch_size"] = config.get("batch_size", "N/A")
            info["learning_rate"] = config.get("learning_rate", "N/A")
            info["max_seq_length"] = config.get("max_seq_length", "N/A")
            info["final_loss"] = config.get("final_loss", "N/A")
            info["training_time"] = config.get("training_time_seconds", 0)
            info["train_samples"] = config.get("train_samples", 0)
        except Exception as e:
            pass
    
    # Try to load adapter config for additional info
    adapter_config = adapter_path / "final_adapter" / "adapter_config.json"
    if adapter_config.exists():
        try:
            with open(adapter_config) as f:
                config = json.load(f)
            info["base_model"] = info.get("base_model") or config.get("base_model_name_or_path", "Unknown")
            info["lora_r"] = info.get("lora_r") or config.get("r", "N/A")
            info["lora_alpha"] = info.get("lora_alpha") or config.get("lora_alpha", "N/A")
        except Exception:
            pass
    
    # Extract timestamp from folder name
    try:
        if adapter_path.name.startswith("run_"):
            timestamp_str = adapter_path.name.replace("run_", "")
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            info["timestamp"] = dt.strftime("%Y-%m-%d %H:%M")
    except:
        info["timestamp"] = "Unknown"
    
    return info


def load_training_configs(adapters: List[Dict], selected_names: List[str]):
    """Load and display training configurations for selected adapters."""
    st.session_state.loaded_configs = {}
    for name in selected_names:
        adapter = next((a for a in adapters if a["name"] == name), None)
        if adapter:
            st.session_state.loaded_configs[name] = adapter
    st.success(f"Loaded configs for {len(selected_names)} models")


def render_training_metrics_comparison():
    """Render training metrics comparison charts."""
    st.subheader("📈 Training Metrics Comparison")
    
    if not st.session_state.compare_models:
        st.info("Select models in the Setup tab first.")
        return
    
    # Load training logs for selected models
    metrics_data = load_training_metrics(st.session_state.compare_models)
    
    if not metrics_data:
        st.warning("No training logs found for selected models. Training logs are saved in TensorBoard format.")
        st.markdown("""
        **To view training metrics:**
        - Training metrics are logged to TensorBoard during training
        - Look for `logs` folders in each adapter directory
        - You can run `tensorboard --logdir models/adapters` to view detailed metrics
        """)
        return
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Loss")
        render_loss_chart(metrics_data)
    
    with col2:
        st.markdown("#### Configuration Comparison")
        render_config_comparison()
    
    # Final metrics comparison
    st.divider()
    st.markdown("#### Final Training Metrics")
    render_final_metrics_table()


def load_training_metrics(model_names: List[str]) -> Dict[str, Any]:
    """Load training metrics from adapter directories."""
    metrics = {}
    adapter_dir = Path("./models/adapters")
    
    for name in model_names:
        model_path = adapter_dir / name
        if model_path.exists():
            # Try to load from training_config.yaml
            config_path = model_path / "training_config.yaml"
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    metrics[name] = {
                        "final_loss": config.get("final_loss", 0),
                        "epochs": config.get("epochs", 0),
                        "total_steps": config.get("total_steps", 0),
                        "training_time": config.get("training_time_seconds", 0),
                        "train_samples": config.get("train_samples", 0),
                    }
                except:
                    pass
    
    return metrics


def render_loss_chart(metrics_data: Dict[str, Any]):
    """Render loss comparison chart."""
    if not metrics_data:
        st.info("No loss data available")
        return
    
    # Create simple bar chart of final losses
    chart_data = []
    for name, data in metrics_data.items():
        if "final_loss" in data and data["final_loss"]:
            chart_data.append({
                "Model": name[:15] + "..." if len(name) > 15 else name,
                "Final Loss": float(data["final_loss"]) if data["final_loss"] else 0
            })
    
    if chart_data:
        df = pd.DataFrame(chart_data)
        st.bar_chart(df.set_index("Model"))
    else:
        st.info("No loss data available for comparison")


def render_config_comparison():
    """Render configuration comparison table."""
    adapter_dir = Path("./models/adapters")
    configs = []
    
    for name in st.session_state.compare_models:
        adapter_path = adapter_dir / name
        info = get_adapter_info(adapter_path) if adapter_path.exists() else {}
        
        configs.append({
            "Model": name[:12] + "..." if len(name) > 12 else name,
            "LoRA r": info.get("lora_r", "N/A"),
            "Alpha": info.get("lora_alpha", "N/A"),
            "LR": f"{info.get('learning_rate', 'N/A'):.0e}" if isinstance(info.get("learning_rate"), float) else "N/A",
            "Epochs": info.get("epochs", "N/A"),
        })
    
    if configs:
        df = pd.DataFrame(configs)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_final_metrics_table():
    """Render final metrics comparison table."""
    adapter_dir = Path("./models/adapters")
    metrics = []
    
    for name in st.session_state.compare_models:
        adapter_path = adapter_dir / name
        info = get_adapter_info(adapter_path) if adapter_path.exists() else {}
        
        final_loss = info.get("final_loss")
        training_time = info.get("training_time", 0)
        
        metrics.append({
            "Model": name,
            "Final Loss": f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else "N/A",
            "Training Time": format_time(training_time),
            "Samples": info.get("train_samples", "N/A"),
            "Steps": info.get("total_steps", "N/A") if hasattr(info, "get") else "N/A",
        })
    
    if metrics:
        df = pd.DataFrame(metrics)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Highlight best model
        if any(isinstance(m.get("Final Loss"), str) and m["Final Loss"] != "N/A" for m in metrics):
            losses = [(m["Model"], float(m["Final Loss"])) for m in metrics 
                     if m["Final Loss"] != "N/A"]
            if losses:
                best_model = min(losses, key=lambda x: x[1])
                st.success(f"🏆 **Best by Training Loss:** {best_model[0]} ({best_model[1]:.4f})")


def render_ab_testing():
    """Render A/B testing interface for comparing model responses."""
    st.subheader("🧪 A/B Response Comparison")
    
    if len(st.session_state.compare_models) < 2:
        st.info("Select at least 2 models in the Setup tab for A/B testing.")
        return
    
    st.markdown("Test the same prompt across multiple models and compare responses.")
    
    # Model selection for A/B test
    col1, col2 = st.columns(2)
    
    with col1:
        model_a = st.selectbox(
            "Model A",
            st.session_state.compare_models,
            key="ab_model_a"
        )
    
    with col2:
        remaining_models = [m for m in st.session_state.compare_models if m != model_a]
        model_b = st.selectbox(
            "Model B",
            remaining_models if remaining_models else st.session_state.compare_models,
            key="ab_model_b"
        )
    
    # Test prompt
    st.markdown("#### Test Prompt")
    test_prompt = st.text_area(
        "Enter a prompt to test:",
        value="Summarize this ServiceNow ticket: INC0012345 - User reports slow VPN connection when working from home. Started 2 days ago. Affecting productivity.",
        height=100
    )
    
    # Base model selection
    base_model = st.text_input(
        "Base Model (HuggingFace ID)",
        value=st.session_state.get("selected_model", "microsoft/phi-2"),
        help="The base model used for fine-tuning"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_test = st.button("🚀 Run A/B Test", type="primary", use_container_width=True)
    
    if run_test:
        with st.spinner("Running A/B test... This may take a moment."):
            results = run_ab_comparison(model_a, model_b, base_model, test_prompt)
            st.session_state.ab_results = results
    
    # Display results
    if "ab_results" in st.session_state and st.session_state.ab_results:
        render_ab_results(st.session_state.ab_results)


def run_ab_comparison(model_a: str, model_b: str, base_model: str, prompt: str) -> Dict[str, Any]:
    """Run A/B comparison between two models."""
    results = {
        "model_a": {"name": model_a, "response": None, "latency": None, "error": None},
        "model_b": {"name": model_b, "response": None, "latency": None, "error": None},
        "prompt": prompt,
    }
    
    adapter_dir = Path("./models/adapters")
    
    for key, model_name in [("model_a", model_a), ("model_b", model_b)]:
        adapter_path = adapter_dir / model_name / "final_adapter"
        
        if not adapter_path.exists():
            results[key]["error"] = f"Adapter not found at {adapter_path}"
            continue
        
        try:
            from core.inference import InferenceEngine
            
            engine = InferenceEngine()
            start_time = time.time()
            
            # Load model with adapter
            engine.load_hf_model_with_adapter(base_model, str(adapter_path))
            
            # Generate response
            response = engine.generate(prompt, max_new_tokens=256, temperature=0.7)
            
            latency = time.time() - start_time
            
            results[key]["response"] = response
            results[key]["latency"] = latency
            
            # Cleanup
            engine.unload()
            
        except Exception as e:
            results[key]["error"] = str(e)
    
    return results


def render_ab_results(results: Dict[str, Any]):
    """Render A/B test results."""
    st.divider()
    st.markdown("#### Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Model A: {results['model_a']['name']}**")
        if results['model_a']['error']:
            st.error(f"Error: {results['model_a']['error']}")
        else:
            st.text_area(
                "Response A",
                value=results['model_a']['response'],
                height=200,
                key="response_a",
                disabled=True
            )
            if results['model_a']['latency']:
                st.caption(f"⏱️ Latency: {results['model_a']['latency']:.2f}s")
    
    with col2:
        st.markdown(f"**Model B: {results['model_b']['name']}**")
        if results['model_b']['error']:
            st.error(f"Error: {results['model_b']['error']}")
        else:
            st.text_area(
                "Response B",
                value=results['model_b']['response'],
                height=200,
                key="response_b",
                disabled=True
            )
            if results['model_b']['latency']:
                st.caption(f"⏱️ Latency: {results['model_b']['latency']:.2f}s")
    
    # Rating interface
    st.divider()
    st.markdown("#### Rate the Responses")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👍 Model A is better", use_container_width=True):
            record_preference(results['model_a']['name'], results['model_b']['name'], "a")
            st.success("Preference recorded!")
    
    with col2:
        if st.button("🤝 Tie / Similar", use_container_width=True):
            record_preference(results['model_a']['name'], results['model_b']['name'], "tie")
            st.info("Recorded as tie")
    
    with col3:
        if st.button("👍 Model B is better", use_container_width=True):
            record_preference(results['model_b']['name'], results['model_a']['name'], "b")
            st.success("Preference recorded!")


def record_preference(winner: str, loser: str, result: str):
    """Record A/B test preference to history."""
    if "ab_history" not in st.session_state:
        st.session_state.ab_history = []
    
    st.session_state.ab_history.append({
        "timestamp": datetime.now().isoformat(),
        "winner": winner if result != "tie" else None,
        "model_a": winner if result == "a" else loser,
        "model_b": loser if result == "a" else winner,
        "result": result,
    })


def render_comparison_summary():
    """Render overall comparison summary and recommendations."""
    st.subheader("📋 Comparison Summary")
    
    if not st.session_state.compare_models:
        st.info("Select models in the Setup tab to see comparison summary.")
        return
    
    # Gather all metrics
    adapter_dir = Path("./models/adapters")
    summary_data = []
    
    for name in st.session_state.compare_models:
        adapter_path = adapter_dir / name
        info = get_adapter_info(adapter_path) if adapter_path.exists() else {}
        
        score = calculate_model_score(info)
        
        summary_data.append({
            "name": name,
            "info": info,
            "score": score,
        })
    
    # Sort by score
    summary_data.sort(key=lambda x: x["score"], reverse=True)
    
    # Display ranking
    st.markdown("#### 🏆 Model Rankings")
    
    for i, model in enumerate(summary_data):
        rank = i + 1
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        
        with st.expander(f"{medal} {model['name']} (Score: {model['score']:.2f})", expanded=(rank == 1)):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Training Config:**")
                info = model['info']
                st.markdown(f"""
                - Base Model: `{info.get('base_model', 'Unknown')}`
                - LoRA Rank: {info.get('lora_r', 'N/A')}
                - Learning Rate: {info.get('learning_rate', 'N/A')}
                - Epochs: {info.get('epochs', 'N/A')}
                """)
            
            with col2:
                st.markdown("**Performance:**")
                final_loss = info.get('final_loss')
                st.markdown(f"""
                - Final Loss: {f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else 'N/A'}
                - Training Time: {format_time(info.get('training_time', 0))}
                - Samples: {info.get('train_samples', 'N/A')}
                """)
    
    # A/B Test History
    if "ab_history" in st.session_state and st.session_state.ab_history:
        st.divider()
        st.markdown("#### 📊 A/B Test History")
        
        # Count wins
        win_counts = {}
        for test in st.session_state.ab_history:
            if test["result"] != "tie" and test["winner"]:
                win_counts[test["winner"]] = win_counts.get(test["winner"], 0) + 1
        
        if win_counts:
            df = pd.DataFrame([
                {"Model": k, "A/B Wins": v} 
                for k, v in sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📥 Export Comparison Report", use_container_width=True):
            export_comparison_report(summary_data)


def calculate_model_score(info: Dict[str, Any]) -> float:
    """Calculate a composite score for model ranking."""
    score = 50.0  # Base score
    
    # Lower loss is better (major factor)
    final_loss = info.get('final_loss')
    if isinstance(final_loss, (int, float)) and final_loss > 0:
        # Typical losses range from 0.1 to 3.0
        # Lower is better, scale to contribute up to 30 points
        loss_score = max(0, 30 - (final_loss * 10))
        score += loss_score
    
    # More training samples is better
    samples = info.get('train_samples', 0)
    if samples and samples > 0:
        sample_score = min(10, samples / 100)  # Up to 10 points
        score += sample_score
    
    # More epochs can be better (up to a point)
    epochs = info.get('epochs', 0)
    if epochs and epochs > 0:
        epoch_score = min(5, epochs)  # Up to 5 points
        score += epoch_score
    
    return score


def export_comparison_report(summary_data: List[Dict]):
    """Export comparison report to JSON."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "models_compared": len(summary_data),
        "rankings": [],
    }
    
    for i, model in enumerate(summary_data):
        report["rankings"].append({
            "rank": i + 1,
            "model": model["name"],
            "score": model["score"],
            "config": model["info"],
        })
    
    # Add A/B history if available
    if "ab_history" in st.session_state:
        report["ab_tests"] = st.session_state.ab_history
    
    # Save to file
    output_path = Path("./logs/comparison_reports")
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename
    
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    st.success(f"Report saved to: {filepath}")
    
    # Also offer download
    st.download_button(
        "📥 Download Report",
        data=json.dumps(report, indent=2, default=str),
        file_name=filename,
        mime="application/json"
    )


def format_time(seconds: float) -> str:
    """Format seconds into readable time string."""
    if not seconds:
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


if __name__ == "__main__":
    render_model_compare()
