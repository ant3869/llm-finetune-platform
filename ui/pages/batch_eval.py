"""
Batch Evaluation UI Page.

Run automated evaluations with metrics tracking.
"""

import streamlit as st
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_batch_evaluation():
    """Render the batch evaluation page."""
    st.title("📈 Batch Evaluation")
    st.markdown("Run automated evaluation with metrics on test datasets.")
    
    # Check if model is loaded
    if not st.session_state.get("eval_model_loaded", False):
        st.warning("⚠️ Please load a model first in the Evaluation page (Step 4)")
        if st.button("← Go to Evaluation"):
            st.session_state.current_step = 4
            st.rerun()
        return
    
    tab1, tab2, tab3 = st.tabs(["🧪 Run Evaluation", "📊 Results", "⚖️ Compare Models"])
    
    with tab1:
        render_run_evaluation()
    
    with tab2:
        render_results_viewer()
    
    with tab3:
        render_model_comparison()


def render_run_evaluation():
    """Render the evaluation runner."""
    st.subheader("Run Batch Evaluation")
    
    # Test data source
    col1, col2 = st.columns(2)
    
    with col1:
        data_source = st.radio(
            "Test Data Source",
            ["Upload File", "Use Template", "Manual Input"]
        )
    
    test_cases = []
    
    if data_source == "Upload File":
        uploaded = st.file_uploader(
            "Upload test file (JSON/JSONL)",
            type=["json", "jsonl"],
            help="Format: {instruction, input, output}"
        )
        
        if uploaded:
            try:
                content = uploaded.read().decode("utf-8")
                if uploaded.name.endswith(".jsonl"):
                    test_cases = [json.loads(line) for line in content.strip().split("\n") if line]
                else:
                    data = json.loads(content)
                    test_cases = data if isinstance(data, list) else [data]
                st.success(f"Loaded {len(test_cases)} test cases")
            except Exception as e:
                st.error(f"Failed to parse file: {e}")
    
    elif data_source == "Use Template":
        with col2:
            template_dir = Path("./data/templates")
            templates = list(template_dir.glob("*.json")) if template_dir.exists() else []
            
            if templates:
                template_names = [t.stem for t in templates]
                selected = st.selectbox("Select Template", template_names)
                template_path = template_dir / f"{selected}.json"
                
                try:
                    with open(template_path) as f:
                        test_cases = json.load(f)
                    st.info(f"Using {len(test_cases)} examples from '{selected}'")
                except Exception as e:
                    st.error(f"Failed to load template: {e}")
            else:
                st.warning("No templates found in data/templates/")
    
    else:  # Manual Input
        st.markdown("Enter test cases (one per line, JSON format):")
        manual_input = st.text_area(
            "Test Cases",
            value='{"instruction": "Summarize this ticket", "input": "User cannot login", "output": "Login issue reported"}',
            height=150,
        )
        
        if manual_input:
            try:
                for line in manual_input.strip().split("\n"):
                    if line.strip():
                        test_cases.append(json.loads(line))
                st.success(f"Parsed {len(test_cases)} test cases")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
    
    st.divider()
    
    # Generation settings
    st.subheader("Generation Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_tokens = st.slider("Max Tokens", 64, 1024, 256)
    with col2:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    with col3:
        eval_name = st.text_input("Evaluation Name", value=f"eval_{len(test_cases)}_samples")
    
    # Run button
    st.divider()
    
    if st.button("🚀 Run Evaluation", type="primary", disabled=len(test_cases) == 0):
        run_batch_evaluation(test_cases, max_tokens, temperature, eval_name)


def run_batch_evaluation(test_cases, max_tokens, temperature, eval_name):
    """Run the batch evaluation."""
    from core.evaluator import BatchEvaluator
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(current, total, status):
        progress_bar.progress(current / total)
        status_text.text(status)
    
    try:
        evaluator = BatchEvaluator(st.session_state.eval_engine)
        evaluator.set_progress_callback(progress_callback)
        
        # Get model name
        model_info = st.session_state.eval_engine.get_model_info()
        model_name = model_info.get("model_path", "unknown")
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        
        # Run evaluation
        results = evaluator.run_evaluation(
            test_cases,
            max_new_tokens=max_tokens,
            temperature=temperature,
            model_name=model_name,
        )
        
        # Save results
        output_dir = Path("./evaluations")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{eval_name}.json"
        results.save(str(output_path))
        
        # Store in session state
        if "evaluation_results" not in st.session_state:
            st.session_state.evaluation_results = []
        st.session_state.evaluation_results.append(results)
        
        st.success(f"✅ Evaluation complete! Results saved to {output_path}")
        
        # Show summary
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", results.total_samples)
        with col2:
            st.metric("Total Time", f"{results.total_time_seconds:.1f}s")
        with col3:
            avg_latency = results.aggregate_metrics.get("avg_latency_ms", 0)
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")
        with col4:
            word_overlap = results.aggregate_metrics.get("avg_word_overlap", 0)
            st.metric("Word Overlap", f"{word_overlap:.2%}")
        
        # Detailed metrics
        st.subheader("Aggregate Metrics")
        metrics_df = {k: [f"{v:.4f}"] for k, v in results.aggregate_metrics.items()}
        st.dataframe(metrics_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def render_results_viewer():
    """View past evaluation results."""
    st.subheader("Evaluation Results")
    
    # Load results from files
    results_dir = Path("./evaluations")
    result_files = list(results_dir.glob("*.json")) if results_dir.exists() else []
    
    if not result_files:
        st.info("No evaluation results found. Run an evaluation first!")
        return
    
    selected_file = st.selectbox(
        "Select Results",
        [f.stem for f in result_files]
    )
    
    if selected_file:
        result_path = results_dir / f"{selected_file}.json"
        
        try:
            with open(result_path) as f:
                results = json.load(f)
            
            # Summary
            st.markdown(f"""
            **Model:** {results['model_name']}  
            **Timestamp:** {results['timestamp']}  
            **Samples:** {results['total_samples']}  
            **Total Time:** {results['total_time_seconds']:.1f}s
            """)
            
            # Metrics
            st.subheader("Aggregate Metrics")
            
            cols = st.columns(4)
            for i, (metric, value) in enumerate(results['aggregate_metrics'].items()):
                with cols[i % 4]:
                    st.metric(metric, f"{value:.4f}")
            
            # Individual results
            st.subheader("Individual Results")
            
            for i, result in enumerate(results['results']):
                with st.expander(f"Sample {i + 1} - Latency: {result['latency_ms']:.0f}ms"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Input:**")
                        st.text(result['input'][:500])
                        
                        st.markdown("**Expected:**")
                        st.text(result['expected'])
                    
                    with col2:
                        st.markdown("**Generated:**")
                        st.text(result['generated'])
                        
                        st.markdown("**Metrics:**")
                        st.json(result['metrics'])
            
        except Exception as e:
            st.error(f"Failed to load results: {e}")


def render_model_comparison():
    """Compare results from different models."""
    st.subheader("Model Comparison")
    
    # Load all results
    results_dir = Path("./evaluations")
    result_files = list(results_dir.glob("*.json")) if results_dir.exists() else []
    
    if len(result_files) < 2:
        st.info("Need at least 2 evaluation results to compare. Run more evaluations!")
        return
    
    # Select results to compare
    selected_files = st.multiselect(
        "Select results to compare",
        [f.stem for f in result_files],
        default=[f.stem for f in result_files[:2]]
    )
    
    if len(selected_files) < 2:
        st.warning("Select at least 2 results to compare")
        return
    
    # Load selected results
    results_data = []
    for name in selected_files:
        path = results_dir / f"{name}.json"
        with open(path) as f:
            results_data.append(json.load(f))
    
    # Comparison table
    st.subheader("Metrics Comparison")
    
    # Get all metrics
    all_metrics = set()
    for r in results_data:
        all_metrics.update(r['aggregate_metrics'].keys())
    
    # Build comparison data
    comparison = {"Metric": []}
    for r in results_data:
        comparison[r['model_name']] = []
    
    for metric in sorted(all_metrics):
        comparison["Metric"].append(metric)
        for r in results_data:
            value = r['aggregate_metrics'].get(metric, 0)
            comparison[r['model_name']].append(f"{value:.4f}")
    
    st.dataframe(comparison, use_container_width=True)
    
    # Winner summary
    st.subheader("Winner by Metric")
    
    for metric in sorted(all_metrics):
        values = [(r['model_name'], r['aggregate_metrics'].get(metric, 0)) for r in results_data]
        
        # Higher is better except for latency
        if "latency" in metric.lower():
            winner = min(values, key=lambda x: x[1])
        else:
            winner = max(values, key=lambda x: x[1])
        
        st.markdown(f"**{metric}:** {winner[0]} ({winner[1]:.4f})")
