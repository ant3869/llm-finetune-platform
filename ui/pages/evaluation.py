"""
Evaluation Page - Step 4

Interactive testing of fine-tuned models with chat interface.
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
    st.markdown("Test your fine-tuned model with interactive chat.")
    
    # Initialize session state for evaluation
    if "eval_model_loaded" not in st.session_state:
        st.session_state.eval_model_loaded = False
    if "eval_engine" not in st.session_state:
        st.session_state.eval_engine = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for model loading
    render_model_loader()
    
    st.divider()
    
    # Main content
    if st.session_state.eval_model_loaded:
        render_chat_interface()
    else:
        render_no_model_state()


def render_model_loader():
    """Render model loading controls in sidebar or main area."""
    st.subheader("🔧 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_source = st.radio(
            "Model Source",
            ["Fine-tuned Adapter", "GGUF Model", "Base Model Only"],
            help="Choose what to load for testing"
        )
    
    with col2:
        if model_source == "Fine-tuned Adapter":
            # Show available adapters
            adapter_dir = Path("./models/adapters")
            adapters = []
            if adapter_dir.exists():
                adapters = [d.name for d in adapter_dir.iterdir() if d.is_dir() and (d / "adapter_config.json").exists()]
            
            if adapters:
                selected_adapter = st.selectbox(
                    "Select Adapter",
                    adapters,
                    help="Choose a trained adapter to test"
                )
                adapter_path = str(adapter_dir / selected_adapter)
            else:
                st.warning("No trained adapters found. Train a model first!")
                adapter_path = None
                
        elif model_source == "GGUF Model":
            # Show GGUF files
            from core.model_loader import ModelLoader
            loader = ModelLoader()
            gguf_models = loader.scan_local_models()
            
            if gguf_models:
                gguf_names = [m["name"] for m in gguf_models]
                selected_gguf = st.selectbox(
                    "Select GGUF Model",
                    gguf_names
                )
                gguf_path = next(m["path"] for m in gguf_models if m["name"] == selected_gguf)
            else:
                st.warning("No GGUF models found in models/base/")
                gguf_path = None
    
    # Base model selection (for adapters)
    if model_source in ["Fine-tuned Adapter", "Base Model Only"]:
        base_model = st.text_input(
            "Base Model",
            value=st.session_state.get("selected_model", "microsoft/phi-2"),
            help="HuggingFace model ID"
        )
    
    # Load button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.session_state.eval_model_loaded:
            if st.button("🔄 Unload Model", type="secondary", use_container_width=True):
                unload_model()
                st.rerun()
        else:
            load_disabled = False
            if model_source == "Fine-tuned Adapter" and not adapters:
                load_disabled = True
            elif model_source == "GGUF Model" and not gguf_models:
                load_disabled = True
            
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
                        st.success("✅ Model loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")
    
    # Show model info if loaded
    if st.session_state.eval_model_loaded and st.session_state.eval_engine:
        info = st.session_state.eval_engine.get_model_info()
        st.info(f"**Loaded:** {info.get('type', 'unknown')} | {info.get('model_path', 'N/A')}")


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
    st.info("👆 Load a model above to start testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What you can test:
        - **Chat responses** - Interactive conversation
        - **IT Support tasks** - Ticket summarization, KB generation
        - **Custom prompts** - Any task you trained for
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Testing Tips:
        - Compare fine-tuned vs base model
        - Test with similar data to training
        - Try edge cases and variations
        """)
    
    # Quick navigation
    st.divider()
    
    if not st.session_state.trained_adapter_path:
        st.warning("No trained adapter found. Complete training first!")
        if st.button("← Go to Training", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()


def render_chat_interface():
    """Render the chat interface for model testing."""
    st.subheader("💬 Chat Interface")
    
    # Generation settings
    with st.expander("⚙️ Generation Settings", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1,
                help="Higher = more creative, Lower = more focused")
        with col2:
            max_tokens = st.slider("Max Tokens", 32, 1024, 256, 32,
                help="Maximum response length")
        with col3:
            top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1,
                help="Nucleus sampling threshold")
        with col4:
            top_k = st.slider("Top K", 1, 100, 50, 1,
                help="Top-k sampling")
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt (optional)",
        value="You are a helpful IT support assistant. Provide clear, concise answers.",
        height=80,
        help="Set the assistant's behavior"
    )
    
    st.divider()
    
    # Chat display
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").write(content)
    
    # Chat input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        with st.spinner("Generating response..."):
            try:
                engine = st.session_state.eval_engine
                
                response = engine.chat(
                    messages=st.session_state.chat_history,
                    system_prompt=system_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                
                # Add assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
                # Remove the user message if generation failed
                st.session_state.chat_history.pop()
        
        st.rerun()
    
    # Chat controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col3:
        if st.button("📋 Copy Last Response", use_container_width=True):
            if st.session_state.chat_history:
                last_assistant = next(
                    (m["content"] for m in reversed(st.session_state.chat_history) 
                     if m["role"] == "assistant"),
                    None
                )
                if last_assistant:
                    st.code(last_assistant, language=None)
    
    st.divider()
    
    # Quick test prompts
    render_quick_prompts()


def render_quick_prompts():
    """Render quick test prompts for IT support tasks."""
    st.subheader("🚀 Quick Test Prompts")
    
    prompts = {
        "Ticket Summary": "Summarize this ticket: User reports VPN connection drops every 10 minutes. They're using Windows 11 and Cisco AnyConnect. Issue started after recent Windows update. User is remote worker and needs VPN for all work tasks.",
        "KB Article": "Write a knowledge article for: How to reset MFA (Multi-Factor Authentication) when user gets a new phone.",
        "Troubleshooting": "What are the troubleshooting steps for: Outlook keeps asking for password repeatedly?",
        "SOP Generation": "Create an SOP for: Onboarding a new employee's IT equipment and accounts.",
    }
    
    cols = st.columns(len(prompts))
    
    for col, (name, prompt) in zip(cols, prompts.items()):
        with col:
            if st.button(name, use_container_width=True, key=f"quick_{name}"):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": prompt
                })
                
                with st.spinner("Generating..."):
                    try:
                        engine = st.session_state.eval_engine
                        response = engine.chat(
                            messages=st.session_state.chat_history,
                            max_new_tokens=512,
                            temperature=0.7,
                        )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.chat_history.pop()
                
                st.rerun()
