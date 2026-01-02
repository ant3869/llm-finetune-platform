"""
Post-Tuning Test Page - Comprehensive Testing & Rating

Test and rate fine-tuned models with:
- Before/After comparison
- Response rating system
- Systematic test scenarios
- Effectiveness dashboard
"""

import streamlit as st
from pathlib import Path
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_post_tuning_test():
    """Render the post-tuning test page."""
    st.title("🧪 Post-Tuning Test Suite")
    st.markdown("Systematically test and rate your fine-tuned model to measure tuning effectiveness.")
    
    # Initialize session state
    init_test_session_state()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 Before/After Compare",
        "📝 Test & Rate",
        "📊 Effectiveness Dashboard",
        "📋 Test History"
    ])
    
    with tab1:
        render_before_after_comparison()
    
    with tab2:
        render_test_and_rate()
    
    with tab3:
        render_effectiveness_dashboard()
    
    with tab4:
        render_test_history()


def init_test_session_state():
    """Initialize session state for testing."""
    defaults = {
        "test_suite_manager": None,
        "base_engine": None,
        "tuned_engine": None,
        "base_model_loaded": False,
        "tuned_model_loaded": False,
        "current_test_responses": {},
        "comparison_history": [],
        "rating_history": [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize test suite manager
    if st.session_state.test_suite_manager is None:
        from core.test_suite import TestSuiteManager
        st.session_state.test_suite_manager = TestSuiteManager()


def render_before_after_comparison():
    """Render before/after comparison interface."""
    st.subheader("🔄 Before/After Comparison")
    st.markdown("Compare base model responses with fine-tuned model responses side-by-side.")
    
    # Model loading section
    with st.expander("🔧 Load Models for Comparison", expanded=not (st.session_state.base_model_loaded and st.session_state.tuned_model_loaded)):
        render_comparison_model_loader()
    
    if not (st.session_state.base_model_loaded and st.session_state.tuned_model_loaded):
        st.info("👆 Load both base and fine-tuned models above to start comparison.")
        return
    
    st.success("✅ Both models loaded! Ready for comparison.")
    
    st.divider()
    
    # Test case selection
    st.subheader("📝 Select Test Case")
    
    manager = st.session_state.test_suite_manager
    categories = manager.get_all_categories()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_category = st.selectbox("Category", categories)
    
    with col2:
        tests_in_category = manager.get_tests_by_category(selected_category)
        test_names = {t.name: t for t in tests_in_category}
        selected_test_name = st.selectbox("Test Case", list(test_names.keys()))
    
    selected_test = test_names[selected_test_name]
    
    # Show test details
    with st.expander("📋 Test Details", expanded=True):
        st.markdown(f"**Category:** {selected_test.category}")
        st.markdown(f"**Difficulty:** {selected_test.difficulty}")
        st.markdown(f"**Expected Qualities:** {', '.join(selected_test.expected_qualities)}")
        st.text_area("Prompt", value=selected_test.prompt, height=150, disabled=True)
        
        if selected_test.reference_output:
            st.text_area("Reference Output", value=selected_test.reference_output, height=100, disabled=True)
    
    # Run comparison
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
            run_comparison_test(selected_test)
    
    # Display comparison results
    if "current_comparison" in st.session_state and st.session_state.current_comparison:
        render_comparison_results(st.session_state.current_comparison)


def render_comparison_model_loader():
    """Render model loader for comparison."""
    col1, col2 = st.columns(2)
    
    # Base model
    with col1:
        st.markdown("#### Base Model (Before Tuning)")
        
        base_model_id = st.text_input(
            "Model ID",
            value=st.session_state.get("selected_model", "microsoft/phi-2"),
            key="base_model_input",
            help="HuggingFace model ID"
        )
        
        if st.session_state.base_model_loaded:
            st.success("✅ Base model loaded")
            if st.button("Unload Base", key="unload_base"):
                unload_base_model()
                st.rerun()
        else:
            if st.button("📥 Load Base Model", key="load_base", type="primary"):
                with st.spinner("Loading base model..."):
                    load_base_model(base_model_id)
                st.rerun()
    
    # Tuned model
    with col2:
        st.markdown("#### Fine-Tuned Model (After Tuning)")
        
        # Get available adapters
        adapter_dir = Path("./models/adapters")
        adapters = []
        if adapter_dir.exists():
            adapters = [d.name for d in adapter_dir.iterdir() 
                       if d.is_dir() and (d / "final_adapter" / "adapter_config.json").exists()]
        
        if adapters:
            selected_adapter = st.selectbox(
                "Select Adapter",
                adapters,
                key="tuned_adapter_select"
            )
            
            tuned_base_model = st.text_input(
                "Base Model for Adapter",
                value=st.session_state.get("selected_model", "microsoft/phi-2"),
                key="tuned_base_model",
                help="Must match the model used for training"
            )
            
            if st.session_state.tuned_model_loaded:
                st.success("✅ Tuned model loaded")
                if st.button("Unload Tuned", key="unload_tuned"):
                    unload_tuned_model()
                    st.rerun()
            else:
                if st.button("📥 Load Tuned Model", key="load_tuned", type="primary"):
                    adapter_path = str(adapter_dir / selected_adapter / "final_adapter")
                    with st.spinner("Loading fine-tuned model..."):
                        load_tuned_model(tuned_base_model, adapter_path, selected_adapter)
                    st.rerun()
        else:
            st.warning("No trained adapters found. Train a model first!")


def load_base_model(model_id: str):
    """Load base model without adapter."""
    try:
        from core.inference import InferenceEngine
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        engine = InferenceEngine()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        engine.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if engine.tokenizer.pad_token is None:
            engine.tokenizer.pad_token = engine.tokenizer.eos_token
        
        engine.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        engine.model.eval()
        engine.model_type = "hf"
        engine.model_path = model_id
        
        st.session_state.base_engine = engine
        st.session_state.base_model_loaded = True
        st.session_state.base_model_name = model_id
        
    except Exception as e:
        st.error(f"Failed to load base model: {e}")


def load_tuned_model(base_model: str, adapter_path: str, adapter_name: str):
    """Load fine-tuned model with adapter."""
    try:
        from core.inference import InferenceEngine
        
        engine = InferenceEngine()
        engine.load_hf_model_with_adapter(base_model, adapter_path)
        
        st.session_state.tuned_engine = engine
        st.session_state.tuned_model_loaded = True
        st.session_state.tuned_adapter_name = adapter_name
        
    except Exception as e:
        st.error(f"Failed to load tuned model: {e}")


def unload_base_model():
    """Unload base model."""
    if st.session_state.base_engine:
        st.session_state.base_engine.unload()
    st.session_state.base_engine = None
    st.session_state.base_model_loaded = False


def unload_tuned_model():
    """Unload tuned model."""
    if st.session_state.tuned_engine:
        st.session_state.tuned_engine.unload()
    st.session_state.tuned_engine = None
    st.session_state.tuned_model_loaded = False


def run_comparison_test(test_case):
    """Run comparison between base and tuned model."""
    from core.test_suite import TestResponse
    
    base_engine = st.session_state.base_engine
    tuned_engine = st.session_state.tuned_engine
    
    results = {"test": test_case, "base": None, "tuned": None}
    
    # Generate base response
    with st.spinner("Generating base model response..."):
        start = time.time()
        try:
            base_response = base_engine.generate(
                prompt=test_case.prompt,
                max_new_tokens=512,
                temperature=0.7,
            )
            base_latency = (time.time() - start) * 1000
            
            results["base"] = TestResponse(
                test_id=test_case.id,
                model_name=st.session_state.get("base_model_name", "unknown"),
                model_type="base",
                adapter_name=None,
                response=base_response,
                latency_ms=base_latency,
            )
        except Exception as e:
            st.error(f"Base model error: {e}")
            return
    
    # Generate tuned response
    with st.spinner("Generating fine-tuned model response..."):
        start = time.time()
        try:
            tuned_response = tuned_engine.generate(
                prompt=test_case.prompt,
                max_new_tokens=512,
                temperature=0.7,
            )
            tuned_latency = (time.time() - start) * 1000
            
            results["tuned"] = TestResponse(
                test_id=test_case.id,
                model_name=st.session_state.get("base_model_name", "unknown"),
                model_type="tuned",
                adapter_name=st.session_state.get("tuned_adapter_name"),
                response=tuned_response,
                latency_ms=tuned_latency,
            )
        except Exception as e:
            st.error(f"Tuned model error: {e}")
            return
    
    st.session_state.current_comparison = results


def render_comparison_results(results: Dict):
    """Render comparison results with rating interface."""
    st.divider()
    st.subheader("📊 Comparison Results")
    
    test_case = results["test"]
    base_resp = results["base"]
    tuned_resp = results["tuned"]
    
    # Side by side responses
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔵 Base Model Response")
        st.text_area(
            "Response",
            value=base_resp.response,
            height=250,
            key="base_response_display",
            disabled=True
        )
        st.caption(f"⏱️ Latency: {base_resp.latency_ms:.0f}ms | 📏 Length: {len(base_resp.response)} chars")
    
    with col2:
        st.markdown("#### 🟢 Fine-Tuned Model Response")
        st.text_area(
            "Response",
            value=tuned_resp.response,
            height=250,
            key="tuned_response_display",
            disabled=True
        )
        st.caption(f"⏱️ Latency: {tuned_resp.latency_ms:.0f}ms | 📏 Length: {len(tuned_resp.response)} chars")
    
    # Rating section
    st.divider()
    st.subheader("⭐ Rate the Responses")
    
    st.markdown("Rate each response on a scale of 1-5 (1=Poor, 5=Excellent)")
    
    # Rating criteria explanation
    with st.expander("📖 Rating Criteria Guide"):
        st.markdown("""
        | Criterion | Description |
        |-----------|-------------|
        | **Relevance** | Does the response address the prompt appropriately? |
        | **Accuracy** | Is the information correct and factual? |
        | **Quality** | Is the response well-written and clear? |
        | **Helpfulness** | Would this response help a user solve their problem? |
        | **Overall** | Your overall impression of the response |
        """)
    
    col1, col2 = st.columns(2)
    
    # Base model ratings
    with col1:
        st.markdown("##### Base Model Ratings")
        base_relevance = st.slider("Relevance", 1, 5, 3, key="base_relevance")
        base_accuracy = st.slider("Accuracy", 1, 5, 3, key="base_accuracy")
        base_quality = st.slider("Quality", 1, 5, 3, key="base_quality")
        base_helpfulness = st.slider("Helpfulness", 1, 5, 3, key="base_helpfulness")
        base_overall = st.slider("Overall", 1, 5, 3, key="base_overall")
        
        base_avg = (base_relevance + base_accuracy + base_quality + base_helpfulness + base_overall) / 5
        st.metric("Average Rating", f"{base_avg:.1f}")
    
    # Tuned model ratings
    with col2:
        st.markdown("##### Fine-Tuned Model Ratings")
        tuned_relevance = st.slider("Relevance", 1, 5, 3, key="tuned_relevance")
        tuned_accuracy = st.slider("Accuracy", 1, 5, 3, key="tuned_accuracy")
        tuned_quality = st.slider("Quality", 1, 5, 3, key="tuned_quality")
        tuned_helpfulness = st.slider("Helpfulness", 1, 5, 3, key="tuned_helpfulness")
        tuned_overall = st.slider("Overall", 1, 5, 3, key="tuned_overall")
        
        tuned_avg = (tuned_relevance + tuned_accuracy + tuned_quality + tuned_helpfulness + tuned_overall) / 5
        st.metric("Average Rating", f"{tuned_avg:.1f}")
    
    # Comparison verdict
    st.divider()
    
    improvement = tuned_avg - base_avg
    
    if improvement > 0.5:
        st.success(f"🎉 **Fine-tuned model is better!** (+{improvement:.1f} improvement)")
        winner = "tuned"
    elif improvement < -0.5:
        st.warning(f"⚠️ **Base model performed better** ({improvement:.1f})")
        winner = "base"
    else:
        st.info(f"🤝 **Results are similar** (difference: {improvement:.1f})")
        winner = "tie"
    
    # Notes
    notes = st.text_area("Notes (optional)", placeholder="Add any observations about the comparison...")
    
    # Save comparison
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("💾 Save Comparison", type="primary", use_container_width=True):
            save_comparison_result(
                results, 
                {
                    "base": {"relevance": base_relevance, "accuracy": base_accuracy, 
                            "quality": base_quality, "helpfulness": base_helpfulness, "overall": base_overall},
                    "tuned": {"relevance": tuned_relevance, "accuracy": tuned_accuracy,
                             "quality": tuned_quality, "helpfulness": tuned_helpfulness, "overall": tuned_overall}
                },
                winner,
                improvement,
                notes
            )
            st.success("✅ Comparison saved!")


def save_comparison_result(results, ratings, winner, improvement, notes):
    """Save comparison result to history."""
    from core.test_suite import ComparisonResult
    
    # Update response ratings
    base_resp = results["base"]
    base_resp.relevance_rating = ratings["base"]["relevance"]
    base_resp.accuracy_rating = ratings["base"]["accuracy"]
    base_resp.quality_rating = ratings["base"]["quality"]
    base_resp.helpfulness_rating = ratings["base"]["helpfulness"]
    base_resp.overall_rating = ratings["base"]["overall"]
    base_resp.notes = notes
    
    tuned_resp = results["tuned"]
    tuned_resp.relevance_rating = ratings["tuned"]["relevance"]
    tuned_resp.accuracy_rating = ratings["tuned"]["accuracy"]
    tuned_resp.quality_rating = ratings["tuned"]["quality"]
    tuned_resp.helpfulness_rating = ratings["tuned"]["helpfulness"]
    tuned_resp.overall_rating = ratings["tuned"]["overall"]
    tuned_resp.notes = notes
    
    # Create comparison result
    comparison = ComparisonResult(
        test_case=results["test"],
        base_response=base_resp,
        tuned_response=tuned_resp,
        winner=winner,
        improvement_score=improvement / 4.0,  # Normalize to -1 to 1
    )
    
    # Save to manager
    manager = st.session_state.test_suite_manager
    manager.record_response(base_resp)
    manager.record_response(tuned_resp)
    manager.record_comparison(comparison)
    
    # Also save to session state for immediate display
    st.session_state.comparison_history.append({
        "test_name": results["test"].name,
        "category": results["test"].category,
        "winner": winner,
        "improvement": improvement,
        "timestamp": datetime.now().isoformat(),
        "base_avg": base_resp.average_rating(),
        "tuned_avg": tuned_resp.average_rating(),
    })


def render_test_and_rate():
    """Render individual test and rate interface."""
    st.subheader("📝 Test & Rate Individual Responses")
    
    # Check if any model is loaded
    eval_engine = st.session_state.get("eval_engine") or st.session_state.get("tuned_engine")
    
    if not eval_engine:
        st.warning("Load a model first using the Evaluation page or Before/After comparison.")
        return
    
    st.markdown("Test the loaded model with custom prompts and rate the responses.")
    
    manager = st.session_state.test_suite_manager
    
    # Test selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        test_source = st.radio(
            "Test Source",
            ["Predefined Tests", "Custom Prompt"],
            horizontal=True
        )
    
    if test_source == "Predefined Tests":
        categories = manager.get_all_categories()
        
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category", categories, key="rate_category")
        with col2:
            tests = manager.get_tests_by_category(category)
            test = st.selectbox("Test", tests, format_func=lambda t: t.name, key="rate_test")
        
        prompt = test.prompt
        test_id = test.id
    else:
        prompt = st.text_area(
            "Enter your test prompt",
            height=150,
            placeholder="Type a prompt to test the model..."
        )
        test_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Generation settings
    with st.expander("⚙️ Generation Settings"):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, key="rate_temp")
            max_tokens = st.slider("Max Tokens", 64, 1024, 256, key="rate_tokens")
        with col2:
            top_p = st.slider("Top P", 0.1, 1.0, 0.9, key="rate_top_p")
    
    # Generate
    if st.button("🚀 Generate Response", type="primary") and prompt:
        with st.spinner("Generating..."):
            start = time.time()
            try:
                response = eval_engine.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                latency = (time.time() - start) * 1000
                
                st.session_state.current_rate_response = {
                    "prompt": prompt,
                    "response": response,
                    "latency": latency,
                    "test_id": test_id,
                }
            except Exception as e:
                st.error(f"Generation failed: {e}")
    
    # Display and rate response
    if "current_rate_response" in st.session_state and st.session_state.current_rate_response:
        resp_data = st.session_state.current_rate_response
        
        st.divider()
        st.markdown("#### Model Response")
        st.text_area("Response", value=resp_data["response"], height=200, disabled=True)
        st.caption(f"⏱️ Latency: {resp_data['latency']:.0f}ms")
        
        # Rating
        st.markdown("#### Rate This Response")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            r_relevance = st.selectbox("Relevance", [1, 2, 3, 4, 5], index=2, key="r_rel")
        with col2:
            r_accuracy = st.selectbox("Accuracy", [1, 2, 3, 4, 5], index=2, key="r_acc")
        with col3:
            r_quality = st.selectbox("Quality", [1, 2, 3, 4, 5], index=2, key="r_qual")
        with col4:
            r_helpfulness = st.selectbox("Helpfulness", [1, 2, 3, 4, 5], index=2, key="r_help")
        with col5:
            r_overall = st.selectbox("Overall", [1, 2, 3, 4, 5], index=2, key="r_overall")
        
        avg_rating = (r_relevance + r_accuracy + r_quality + r_helpfulness + r_overall) / 5
        st.metric("Average Rating", f"{avg_rating:.1f} / 5.0")
        
        notes = st.text_input("Notes (optional)", key="rate_notes")
        
        if st.button("💾 Save Rating"):
            save_rating(resp_data, {
                "relevance": r_relevance,
                "accuracy": r_accuracy,
                "quality": r_quality,
                "helpfulness": r_helpfulness,
                "overall": r_overall,
            }, notes)
            st.success("✅ Rating saved!")


def save_rating(resp_data, ratings, notes):
    """Save individual rating."""
    from core.test_suite import TestResponse
    
    model_type = "tuned" if st.session_state.get("tuned_model_loaded") else "base"
    
    response = TestResponse(
        test_id=resp_data["test_id"],
        model_name=st.session_state.get("selected_model", "unknown"),
        model_type=model_type,
        adapter_name=st.session_state.get("tuned_adapter_name"),
        response=resp_data["response"],
        latency_ms=resp_data["latency"],
        relevance_rating=ratings["relevance"],
        accuracy_rating=ratings["accuracy"],
        quality_rating=ratings["quality"],
        helpfulness_rating=ratings["helpfulness"],
        overall_rating=ratings["overall"],
        notes=notes,
    )
    
    manager = st.session_state.test_suite_manager
    manager.record_response(response)
    
    st.session_state.rating_history.append({
        "test_id": resp_data["test_id"],
        "model_type": model_type,
        "average_rating": response.average_rating(),
        "timestamp": datetime.now().isoformat(),
    })


def render_effectiveness_dashboard():
    """Render effectiveness metrics dashboard."""
    st.subheader("📊 Tuning Effectiveness Dashboard")
    
    manager = st.session_state.test_suite_manager
    
    # Get metrics
    effectiveness = manager.get_tuning_effectiveness_metrics()
    rating_summary = manager.get_rating_summary()
    
    if not effectiveness and not st.session_state.comparison_history:
        st.info("No comparison data yet. Run some before/after comparisons to see metrics here.")
        
        # Show sample metrics structure
        st.markdown("""
        **After running comparisons, you'll see:**
        - Win rate: How often the tuned model beats the base model
        - Improvement score: Average rating improvement
        - Category breakdown: Performance by test category
        - Rating trends: Overall quality metrics
        """)
        return
    
    # Summary metrics
    st.markdown("### 📈 Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_comparisons = effectiveness.get("total_comparisons", len(st.session_state.comparison_history))
    tuned_wins = effectiveness.get("tuned_wins", sum(1 for c in st.session_state.comparison_history if c["winner"] == "tuned"))
    base_wins = effectiveness.get("base_wins", sum(1 for c in st.session_state.comparison_history if c["winner"] == "base"))
    
    with col1:
        st.metric("Total Comparisons", total_comparisons)
    
    with col2:
        win_rate = (tuned_wins / total_comparisons * 100) if total_comparisons > 0 else 0
        st.metric("Tuned Win Rate", f"{win_rate:.0f}%", 
                 delta=f"+{tuned_wins - base_wins}" if tuned_wins > base_wins else f"{tuned_wins - base_wins}")
    
    with col3:
        avg_improvement = effectiveness.get("avg_improvement_score", 0)
        if not avg_improvement and st.session_state.comparison_history:
            avg_improvement = sum(c["improvement"] for c in st.session_state.comparison_history) / len(st.session_state.comparison_history)
        st.metric("Avg Improvement", f"{avg_improvement:+.2f}")
    
    with col4:
        ties = effectiveness.get("ties", sum(1 for c in st.session_state.comparison_history if c["winner"] == "tie"))
        st.metric("Ties", ties)
    
    # Win/Loss chart
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Comparison Results")
        
        if st.session_state.comparison_history:
            results_df = pd.DataFrame([
                {"Result": "Tuned Better", "Count": tuned_wins},
                {"Result": "Base Better", "Count": base_wins},
                {"Result": "Tie", "Count": ties},
            ])
            st.bar_chart(results_df.set_index("Result"))
    
    with col2:
        st.markdown("### 📊 Category Breakdown")
        
        # Calculate per-category stats
        category_stats = {}
        for comp in st.session_state.comparison_history:
            cat = comp.get("category", "Unknown")
            if cat not in category_stats:
                category_stats[cat] = {"tuned_wins": 0, "total": 0, "improvements": []}
            
            category_stats[cat]["total"] += 1
            if comp["winner"] == "tuned":
                category_stats[cat]["tuned_wins"] += 1
            category_stats[cat]["improvements"].append(comp["improvement"])
        
        if category_stats:
            cat_data = []
            for cat, stats in category_stats.items():
                cat_data.append({
                    "Category": cat,
                    "Win Rate": f"{stats['tuned_wins']/stats['total']*100:.0f}%" if stats['total'] > 0 else "N/A",
                    "Avg Improvement": f"{sum(stats['improvements'])/len(stats['improvements']):+.2f}" if stats['improvements'] else "N/A",
                    "Tests": stats['total'],
                })
            
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)
    
    # Rating trends
    st.divider()
    st.markdown("### ⭐ Rating Summary")
    
    if rating_summary.get("base_model") or rating_summary.get("tuned_model"):
        col1, col2 = st.columns(2)
        
        with col1:
            if rating_summary.get("base_model"):
                stats = rating_summary["base_model"]
                st.markdown("**Base Model Ratings:**")
                st.markdown(f"- Average: {stats['average']:.2f}")
                st.markdown(f"- Range: {stats['min']:.1f} - {stats['max']:.1f}")
                st.markdown(f"- Samples: {stats['count']}")
        
        with col2:
            if rating_summary.get("tuned_model"):
                stats = rating_summary["tuned_model"]
                st.markdown("**Tuned Model Ratings:**")
                st.markdown(f"- Average: {stats['average']:.2f}")
                st.markdown(f"- Range: {stats['min']:.1f} - {stats['max']:.1f}")
                st.markdown(f"- Samples: {stats['count']}")
    
    # Export
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Results", use_container_width=True):
            filepath = manager.save_results()
            st.success(f"Results saved to {filepath}")
    
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.comparison_history = []
            st.session_state.rating_history = []
            st.session_state.test_suite_manager = None
            init_test_session_state()
            st.rerun()


def render_test_history():
    """Render test history."""
    st.subheader("📋 Test History")
    
    # Comparison history
    if st.session_state.comparison_history:
        st.markdown("### 🔄 Comparison History")
        
        history_data = []
        for comp in st.session_state.comparison_history:
            history_data.append({
                "Test": comp["test_name"],
                "Category": comp["category"],
                "Winner": "🟢 Tuned" if comp["winner"] == "tuned" else "🔵 Base" if comp["winner"] == "base" else "🤝 Tie",
                "Improvement": f"{comp['improvement']:+.1f}",
                "Base Avg": f"{comp['base_avg']:.1f}" if comp.get('base_avg') else "N/A",
                "Tuned Avg": f"{comp['tuned_avg']:.1f}" if comp.get('tuned_avg') else "N/A",
                "Time": comp["timestamp"][:19],
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No comparisons recorded yet.")
    
    # Rating history
    if st.session_state.rating_history:
        st.divider()
        st.markdown("### ⭐ Rating History")
        
        rating_data = []
        for rating in st.session_state.rating_history:
            rating_data.append({
                "Test ID": rating["test_id"][:20] + "..." if len(rating["test_id"]) > 20 else rating["test_id"],
                "Model": "🟢 Tuned" if rating["model_type"] == "tuned" else "🔵 Base",
                "Rating": f"{rating['average_rating']:.1f}" if rating.get('average_rating') else "N/A",
                "Time": rating["timestamp"][:19],
            })
        
        df = pd.DataFrame(rating_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Load previous results
    st.divider()
    st.markdown("### 📂 Load Previous Results")
    
    results_dir = Path("./logs/test_results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json"))
        
        if result_files:
            selected_file = st.selectbox(
                "Select results file",
                result_files,
                format_func=lambda x: x.name
            )
            
            if st.button("📥 Load Results"):
                manager = st.session_state.test_suite_manager
                manager.load_results(str(selected_file))
                st.success("Results loaded!")
                st.rerun()
        else:
            st.info("No saved results found.")


if __name__ == "__main__":
    render_post_tuning_test()
