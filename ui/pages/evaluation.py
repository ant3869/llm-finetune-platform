"""
Evaluation Page - Step 4

Interactive testing of fine-tuned models with chat interface
and batch evaluation with metrics.
"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_evaluation():
    """Render the evaluation/testing page."""
    st.title("📊 Step 4: Evaluation & Testing")
    
    # Initialize session state for evaluation
    if "eval_model_loaded" not in st.session_state:
        st.session_state.eval_model_loaded = False
    if "eval_engine" not in st.session_state:
        st.session_state.eval_engine = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # =========================================================================
    # PAGE INTRO
    # =========================================================================
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d1b4e 100%); 
                padding: 1rem 1.25rem; border-radius: 8px; margin-bottom: 1rem;
                border-left: 4px solid #60a5fa;">
        <div style="color: #93c5fd; font-weight: 600; margin-bottom: 0.25rem;">🎯 What This Page Does</div>
        <div style="color: #e0e0e0; font-size: 0.9rem;">
            Test your fine-tuned model interactively or run batch evaluations to measure quality.
            Compare responses, check metrics like BLEU/ROUGE, and validate your training worked.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # MODEL LOADER - Compact Card
    # =========================================================================
    render_model_loader()
    
    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    if st.session_state.eval_model_loaded:
        st.markdown("---")
        tab1, tab2 = st.tabs(["💬 Interactive Chat", "📈 Batch Evaluation"])
        
        with tab1:
            render_chat_interface()
        
        with tab2:
            render_batch_evaluation()
    else:
        render_no_model_state()


def render_model_loader():
    """Render compact model loading controls."""
    
    st.markdown("""
    <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; 
                padding: 0.75rem 1rem; margin-bottom: 1rem;">
        <div style="color: #60a5fa; font-weight: 600; font-size: 1rem;">🔧 Model Configuration</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Use columns for compact layout
    col_left, col_right = st.columns([1, 1], gap="medium")
    
    # Initialize variables
    adapter_path = None
    gguf_path = None
    adapters = []
    gguf_models = []
    base_model = st.session_state.get("selected_model", "microsoft/phi-2")
    
    with col_left:
        # Model source selection
        model_source = st.radio(
            "Model Source",
            ["Fine-tuned Adapter", "GGUF Model", "Base Model Only"],
            help="Choose what to load for testing",
            horizontal=False
        )
        
        # Source-specific selection
        if model_source == "Fine-tuned Adapter":
            adapter_dir = Path("./models/adapters")
            if adapter_dir.exists():
                adapters = [d.name for d in adapter_dir.iterdir() 
                           if d.is_dir() and (d / "adapter_config.json").exists()]
            
            if adapters:
                selected_adapter = st.selectbox(
                    "Trained Adapter",
                    adapters,
                    help="Select from your trained adapters"
                )
                adapter_path = str(adapter_dir / selected_adapter)
            else:
                st.warning("⚠️ No trained adapters found")
                
        elif model_source == "GGUF Model":
            from core.model_loader import ModelLoader
            loader = ModelLoader()
            gguf_models = loader.scan_models()
            
            if gguf_models:
                gguf_names = [m.name for m in gguf_models]
                selected_gguf = st.selectbox("GGUF Model", gguf_names)
                gguf_path = next(str(m.path) for m in gguf_models if m.name == selected_gguf)
            else:
                st.warning("⚠️ No GGUF models found")
    
    with col_right:
        # Base model input (for HF models)
        if model_source in ["Fine-tuned Adapter", "Base Model Only"]:
            base_model = st.text_input(
                "Base Model ID",
                value=base_model,
                help="HuggingFace model ID (e.g., microsoft/phi-2)"
            )
        
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        
        # Load/Unload button
        if st.session_state.eval_model_loaded:
            # Show current model info
            if st.session_state.eval_engine:
                info = st.session_state.eval_engine.get_model_info()
                st.success(f"✅ **Loaded:** {info.get('model_path', 'N/A')}")
            
            if st.button("🔄 Unload Model", type="secondary", use_container_width=True):
                unload_model()
                st.rerun()
        else:
            # Determine if we can load
            load_disabled = False
            load_reason = ""
            
            if model_source == "Fine-tuned Adapter":
                if not adapters:
                    load_disabled = True
                    load_reason = "Train a model first"
                elif not adapter_path:
                    load_disabled = True
                    load_reason = "Select an adapter"
            elif model_source == "GGUF Model":
                if not gguf_models:
                    load_disabled = True
                    load_reason = "No GGUF models found"
            
            if load_disabled:
                st.caption(f"💡 {load_reason}")
            
            if st.button("📥 Load Model", type="primary", use_container_width=True, disabled=load_disabled):
                with st.spinner("Loading model... This may take a few minutes."):
                    try:
                        if model_source == "Fine-tuned Adapter":
                            load_hf_with_adapter(base_model, adapter_path)
                        elif model_source == "GGUF Model":
                            load_gguf(gguf_path)
                        else:
                            load_hf_base(base_model)
                        
                        st.session_state.eval_model_loaded = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")


def load_hf_with_adapter(base_model: str, adapter_path: str):
    """Load HuggingFace model with LoRA adapter."""
    from core.inference import InferenceEngine
    
    engine = InferenceEngine()
    engine.load_hf_model_with_adapter(base_model, adapter_path)
    st.session_state.eval_engine = engine


def load_gguf(gguf_path: str):
    """Load GGUF model."""
    from core.inference import InferenceEngine
    
    engine = InferenceEngine()
    engine.load_gguf_model(gguf_path)
    st.session_state.eval_engine = engine


def load_hf_base(base_model: str):
    """Load base HuggingFace model without adapter."""
    from core.inference import InferenceEngine
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    
    engine = InferenceEngine()
    
    # Load directly without adapter
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    engine.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if engine.tokenizer.pad_token is None:
        engine.tokenizer.pad_token = engine.tokenizer.eos_token
    
    engine.model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    engine.model.eval()
    engine.model_type = "hf"
    engine.model_path = base_model
    
    st.session_state.eval_engine = engine


def unload_model():
    """Unload the current model."""
    if st.session_state.eval_engine:
        st.session_state.eval_engine.unload()
        st.session_state.eval_engine = None
    st.session_state.eval_model_loaded = False
    st.session_state.chat_history = []


def render_no_model_state():
    """Render state when no model is loaded."""
    
    st.markdown("---")
    
    # Helpful guidance
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("""
        <div style="background: #1a2332; border: 1px solid #2d4a6f; border-radius: 8px; padding: 1rem;">
            <div style="color: #60a5fa; font-weight: 600; margin-bottom: 0.5rem;">🎯 What You Can Test</div>
            <ul style="color: #b0b0b0; margin: 0; padding-left: 1.25rem; font-size: 0.9rem;">
                <li><strong>Chat responses</strong> — Interactive conversation</li>
                <li><strong>IT Support tasks</strong> — Ticket summaries, KB articles</li>
                <li><strong>Custom prompts</strong> — Any task you trained for</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #2a2215; border: 1px solid #5c4a1f; border-radius: 8px; padding: 1rem;">
            <div style="color: #fbbf24; font-weight: 600; margin-bottom: 0.5rem;">💡 Testing Tips</div>
            <ul style="color: #b0b0b0; margin: 0; padding-left: 1.25rem; font-size: 0.9rem;">
                <li>Compare fine-tuned vs base model</li>
                <li>Test with similar data to training</li>
                <li>Try edge cases and variations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    
    # Quick navigation if no adapter
    if not st.session_state.get("trained_adapter_path"):
        st.info("👆 **Load a model above to start testing.** Select an adapter from your training runs, a GGUF model, or test the base model.")


def render_chat_interface():
    """Render the chat interface for model testing."""
    
    # Two-column layout: settings on left, chat on right
    settings_col, chat_col = st.columns([1, 2], gap="medium")
    
    with settings_col:
        # Generation settings card
        st.markdown("""
        <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="color: #60a5fa; font-weight: 600; font-size: 0.9rem;">⚙️ Generation Settings</div>
        </div>
        """, unsafe_allow_html=True)
        
        temperature = st.slider(
            "Temperature",
            0.0, 2.0, 0.7, 0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            32, 1024, 256, 32,
            help="Maximum response length"
        )
        
        with st.expander("Advanced", expanded=False):
            top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1, help="Nucleus sampling")
            top_k = st.slider("Top K", 1, 100, 50, 1, help="Top-k sampling")
        
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        
        # System prompt
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful IT support assistant. Provide clear, concise answers.",
            height=100,
            help="Set the assistant's behavior"
        )
        
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        
        # Quick test prompts in a compact card
        st.markdown("""
        <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem;">
            <div style="color: #f59e0b; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;">🚀 Quick Tests</div>
        </div>
        """, unsafe_allow_html=True)
        
        quick_prompts = {
            "📋 Ticket Summary": "Summarize this ticket: User reports VPN connection drops every 10 minutes. Using Windows 11 with Cisco AnyConnect. Started after recent Windows update. Remote worker needs VPN for all tasks.",
            "📚 KB Article": "Write a knowledge article for: How to reset MFA when user gets a new phone.",
            "🔧 Troubleshoot": "Troubleshooting steps for: Outlook keeps asking for password repeatedly?",
            "📝 SOP": "Create an SOP for: Onboarding a new employee's IT equipment and accounts.",
        }
        
        for name, prompt in quick_prompts.items():
            if st.button(name, use_container_width=True, key=f"quick_{name}"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.spinner("Generating..."):
                    try:
                        engine = st.session_state.eval_engine
                        response = engine.chat(
                            messages=st.session_state.chat_history,
                            system_prompt=system_prompt,
                            max_new_tokens=512,
                            temperature=0.7,
                        )
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.chat_history.pop()
                st.rerun()
    
    with chat_col:
        # Chat container
        st.markdown("""
        <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
            <div style="color: #60a5fa; font-weight: 600; font-size: 0.9rem;">💬 Chat</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat display area
        chat_container = st.container(height=400)
        
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="color: #6b7280; text-align: center; padding: 2rem; font-style: italic;">
                    Type a message below or use a quick test prompt ←
                </div>
                """, unsafe_allow_html=True)
            else:
                for message in st.session_state.chat_history:
                    role = message["role"]
                    content = message["content"]
                    st.chat_message(role).write(content)
        
        # Chat input
        user_input = st.chat_input("Type your message...")
        
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("Generating response..."):
                try:
                    engine = st.session_state.eval_engine
                    response = engine.chat(
                        messages=st.session_state.chat_history,
                        system_prompt=system_prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.session_state.chat_history.pop()
            
            st.rerun()
        
        # Chat controls
        ctrl_col1, ctrl_col2 = st.columns(2)
        
        with ctrl_col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with ctrl_col2:
            if st.button("📋 Copy Last Response", use_container_width=True):
                if st.session_state.chat_history:
                    last_assistant = next(
                        (m["content"] for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"),
                        None
                    )
                    if last_assistant:
                        st.code(last_assistant, language=None)


def render_batch_evaluation():
    """Render batch evaluation interface with metrics."""
    
    # Two-column layout
    config_col, results_col = st.columns([1, 2], gap="medium")
    
    with config_col:
        # Test Data Source card
        st.markdown("""
        <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="color: #60a5fa; font-weight: 600; font-size: 0.9rem;">📁 Test Data Source</div>
        </div>
        """, unsafe_allow_html=True)
        
        test_source = st.radio(
            "Source",
            ["Training Data", "Upload File", "Templates"],
            label_visibility="collapsed",
            horizontal=False
        )
        
        test_samples = []
        
        if test_source == "Training Data":
            if st.session_state.get("training_samples"):
                samples = st.session_state.training_samples
                st.success(f"✅ {len(samples)} samples available")
                
                max_samples = min(len(samples), 50)
                num_samples = st.slider("Samples to evaluate", 1, max_samples, min(10, max_samples))
                test_samples = samples[:num_samples]
            else:
                st.warning("No training data loaded")
                st.caption("Go to Step 1 to load data")
                
        elif test_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Test dataset",
                type=["json", "jsonl"],
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                try:
                    from core.dataset_handler import DatasetHandler
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    handler = DatasetHandler()
                    test_samples = handler.load_file(Path(tmp_path))
                    st.success(f"✅ {len(test_samples)} samples loaded")
                except Exception as e:
                    st.error(f"Failed: {e}")
                    
        elif test_source == "Templates":
            templates_dir = Path("./data/templates")
            
            if templates_dir.exists():
                template_files = list(templates_dir.glob("*.json"))
                
                if template_files:
                    selected_template = st.selectbox(
                        "Template",
                        [f.stem for f in template_files],
                        format_func=lambda x: x.replace("_", " ").title(),
                        label_visibility="collapsed"
                    )
                    
                    try:
                        from core.dataset_handler import DatasetHandler
                        handler = DatasetHandler()
                        test_samples = handler.load_file(templates_dir / f"{selected_template}.json")
                        st.info(f"📄 {len(test_samples)} samples")
                    except Exception as e:
                        st.error(f"Failed: {e}")
        
        st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
        
        # Evaluation Settings card
        st.markdown("""
        <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="color: #60a5fa; font-weight: 600; font-size: 0.9rem;">⚙️ Evaluation Settings</div>
        </div>
        """, unsafe_allow_html=True)
        
        max_tokens = st.slider("Max Tokens", 64, 512, 256, 32, key="batch_max_tokens")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.1, key="batch_temp",
            help="Lower = more deterministic")
        
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        
        # Metrics selection
        st.markdown("""
        <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="color: #60a5fa; font-weight: 600; font-size: 0.9rem;">📊 Metrics</div>
        </div>
        """, unsafe_allow_html=True)
        
        calc_bleu = st.checkbox("BLEU Score", value=True, help="Translation quality metric")
        calc_rouge = st.checkbox("ROUGE Score", value=True, help="Summary quality metric")
        calc_word_overlap = st.checkbox("Word Overlap", value=True, help="Simple overlap measure")
        
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        
        # Run button
        if test_samples:
            if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
                run_batch_evaluation(
                    test_samples,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    metrics={"bleu": calc_bleu, "rouge": calc_rouge, "word_overlap": calc_word_overlap}
                )
        else:
            st.button("🚀 Run Evaluation", type="primary", use_container_width=True, disabled=True)
            st.caption("Select test data to enable")
    
    with results_col:
        # Results area
        st.markdown("""
        <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="color: #60a5fa; font-weight: 600; font-size: 0.9rem;">📈 Results</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show previous results if available
        if st.session_state.get("batch_eval_results"):
            display_batch_results(st.session_state.batch_eval_results)
        else:
            st.markdown("""
            <div style="color: #6b7280; text-align: center; padding: 3rem; font-style: italic;">
                Results will appear here after running evaluation
            </div>
            """, unsafe_allow_html=True)


def display_batch_results(results):
    """Display batch evaluation results."""
    import pandas as pd
    
    # Summary metrics
    metric_names = [k for k in results[0].keys() if k not in ["Instruction", "Reference", "Generated"]]
    
    if metric_names:
        cols = st.columns(len(metric_names))
        for i, metric_name in enumerate(metric_names):
            values = [r[metric_name] for r in results if isinstance(r.get(metric_name), (int, float))]
            if values:
                avg_value = sum(values) / len(values)
                with cols[i]:
                    st.metric(f"Avg {metric_name}", f"{avg_value:.3f}")
    
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
    
    # Results table
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True, hide_index=True, height=300)
    
    # Export buttons
    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button("📥 CSV", csv_data, "eval_results.csv", "text/csv", use_container_width=True)
    with col2:
        json_data = df.to_json(orient="records", indent=2)
        st.download_button("📥 JSON", json_data, "eval_results.json", "application/json", use_container_width=True)


def run_batch_evaluation(test_samples, max_tokens: int, temperature: float, metrics: dict):
    """Run batch evaluation and display results."""
    import pandas as pd
    
    try:
        from core.evaluator import BatchEvaluator, MetricsCalculator
        
        engine = st.session_state.eval_engine
        evaluator = BatchEvaluator(engine)
        calculator = MetricsCalculator()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total = len(test_samples)
        
        for i, sample in enumerate(test_samples):
            status_text.text(f"Evaluating sample {i+1}/{total}...")
            progress_bar.progress((i + 1) / total)
            
            # Create prompt from sample
            prompt = sample.format_prompt("alpaca").split("### Response:")[0] + "### Response:"
            
            # Generate response
            try:
                generated = engine.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                # Clean up response
                generated = generated.replace(prompt, "").strip()
            except Exception as e:
                generated = f"[Error: {e}]"
            
            # Calculate metrics
            reference = sample.output
            sample_metrics = {}
            
            if metrics.get("bleu"):
                sample_metrics["BLEU"] = calculator.bleu_score(generated, reference)
            if metrics.get("rouge"):
                rouge_scores = calculator.rouge_scores(generated, reference)
                sample_metrics["ROUGE-1"] = rouge_scores.get("rouge-1", 0)
                sample_metrics["ROUGE-L"] = rouge_scores.get("rouge-l", 0)
            if metrics.get("word_overlap"):
                sample_metrics["Word Overlap"] = calculator.word_overlap(generated, reference)
            
            results.append({
                "Instruction": sample.instruction[:80] + "..." if len(sample.instruction) > 80 else sample.instruction,
                "Reference": reference[:100] + "..." if len(reference) > 100 else reference,
                "Generated": generated[:100] + "..." if len(generated) > 100 else generated,
                **sample_metrics
            })
        
        progress_bar.empty()
        status_text.empty()
        
        # Store and display results
        st.session_state.batch_eval_results = results
        st.success(f"✅ Evaluated {total} samples!")
        st.rerun()
        
    except ImportError as e:
        st.error(f"Missing evaluation dependencies: {e}")
        st.info("Install with: pip install nltk rouge-score")
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        import traceback
        st.code(traceback.format_exc())
