"""
Model Selection Page - Step 2

Select HuggingFace model for fine-tuning and configure settings.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_loader import ModelLoader


# Popular models for fine-tuning on consumer hardware
RECOMMENDED_MODELS = [
    {
        "name": "Microsoft Phi-2",
        "id": "microsoft/phi-2",
        "size": "2.7B",
        "vram": "~3GB",
        "description": "Excellent small model, great for testing",
        "recommended": True,
    },
    {
        "name": "Microsoft Phi-3 Mini",
        "id": "microsoft/phi-3-mini-4k-instruct",
        "size": "3.8B",
        "vram": "~4GB",
        "description": "Improved Phi model with instruction tuning",
        "recommended": True,
    },
    {
        "name": "IBM Granite 3.0 8B",
        "id": "ibm-granite/granite-3.0-8b-instruct",
        "size": "8B",
        "vram": "~5GB",
        "description": "Enterprise-grade model for IT tasks",
        "recommended": True,
    },
    {
        "name": "Mistral 7B",
        "id": "mistralai/Mistral-7B-v0.1",
        "size": "7B",
        "vram": "~5GB",
        "description": "Popular open-source model",
        "recommended": False,
    },
    {
        "name": "Llama 2 7B",
        "id": "meta-llama/Llama-2-7b-hf",
        "size": "7B",
        "vram": "~5GB",
        "description": "Meta's foundation model (requires access)",
        "recommended": False,
    },
    {
        "name": "TinyLlama 1.1B",
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "size": "1.1B",
        "vram": "~2GB",
        "description": "Very small, fast training for quick tests",
        "recommended": False,
    },
]


def render_model_select():
    """Render the model selection page."""
    st.title("🤖 Step 2: Model Selection")
    st.markdown("Choose a base model for fine-tuning. Models will be loaded with 4-bit quantization to fit in 8GB VRAM.")
    
    # Check prerequisites
    if not st.session_state.training_samples:
        st.warning("⚠️ Please load training data first (Step 1)")
        if st.button("← Back to Data Preparation"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    # GPU status
    render_gpu_status()
    
    st.divider()
    
    # Model selection tabs
    tab1, tab2, tab3 = st.tabs(["🌟 Recommended Models", "📁 Local GGUF Models", "✏️ Custom Model"])
    
    with tab1:
        render_recommended_models()
    
    with tab2:
        render_local_gguf_models()
    
    with tab3:
        render_custom_model()
    
    # Show selected model info
    if st.session_state.selected_model:
        st.divider()
        render_selected_model_info()
        
        # Navigation
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Back to Data", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        
        with col3:
            if st.button("Next: Training →", type="primary", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()


def render_gpu_status():
    """Render GPU status and VRAM estimation."""
    st.subheader("💾 Hardware Status")
    
    col1, col2, col3 = st.columns(3)
    
    loader = ModelLoader()
    gpu_info = loader.check_gpu_availability()
    
    with col1:
        if gpu_info["cuda_available"] and gpu_info["devices"]:
            device = gpu_info["devices"][0]
            st.metric("GPU", device["name"][:25])
        else:
            st.metric("GPU", "Not Available")
            st.warning("Training without GPU will be very slow!")
    
    with col2:
        if gpu_info["cuda_available"] and gpu_info["devices"]:
            device = gpu_info["devices"][0]
            st.metric("Total VRAM", f"{device['total_memory_gb']:.1f} GB")
        else:
            st.metric("Total VRAM", "N/A")
    
    with col3:
        if gpu_info["cuda_available"] and gpu_info["devices"]:
            device = gpu_info["devices"][0]
            st.metric("Free VRAM", f"{device['free_memory_gb']:.1f} GB")
        else:
            st.metric("Free VRAM", "N/A")
    
    # Show error if any
    if gpu_info.get("error"):
        st.warning(f"⚠️ {gpu_info['error']}")


def render_local_gguf_models():
    """Render local GGUF model selection with file browser."""
    st.subheader("📁 Local GGUF Models")
    
    st.markdown("""
    **Note:** GGUF files can't be directly fine-tuned. When you select a GGUF file, 
    you'll need to specify the corresponding HuggingFace model for training.
    After training, your LoRA adapter can be used with your GGUF file for inference.
    """)
    
    # Scan default models directory
    loader = ModelLoader()
    models_dir = Path("./models/base")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Custom directory input
        custom_dir = st.text_input(
            "Models Directory",
            value=str(models_dir.absolute()),
            help="Path to folder containing GGUF files"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_btn = st.button("🔍 Scan Directory", use_container_width=True)
    
    # Scan for models
    scan_path = Path(custom_dir) if custom_dir else models_dir
    
    if scan_btn or "scanned_models" not in st.session_state:
        try:
            found_models = loader.scan_models(scan_path)
            st.session_state.scanned_models = found_models
        except Exception as e:
            st.error(f"Error scanning directory: {e}")
            st.session_state.scanned_models = []
    
    found_models = st.session_state.get("scanned_models", [])
    
    if found_models:
        st.success(f"Found {len(found_models)} GGUF model(s)")
        
        for model in found_models:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{model.name}**")
                    st.caption(f"📁 {model.path}")
                
                with col2:
                    st.metric("Size", f"{model.size_gb:.1f} GB")
                
                with col3:
                    if st.button("Select", key=f"gguf_{model.name}", use_container_width=True):
                        # Store GGUF path and prompt for HF model
                        st.session_state.selected_gguf = str(model.path)
                        st.session_state.show_hf_mapping = True
                        st.rerun()
        
        st.divider()
    else:
        st.info(f"No GGUF files found in: {scan_path}")
        st.markdown("""
        **To use local models:**
        1. Place your `.gguf` files in the `models/base/` folder, or
        2. Enter the path to your models folder above and click "Scan Directory"
        """)
    
    # Show HF model mapping dialog
    if st.session_state.get("show_hf_mapping"):
        st.markdown("---")
        st.subheader("🔗 Select Training Model")
        
        gguf_path = st.session_state.get("selected_gguf", "")
        gguf_name = Path(gguf_path).stem if gguf_path else ""
        
        st.info(f"Selected GGUF: **{gguf_name}**")
        
        st.markdown("""
        Since GGUF files can't be directly fine-tuned, select the corresponding 
        HuggingFace model to use for training:
        """)
        
        # Try to auto-detect HF model
        suggested_hf = loader.get_hf_model_id(gguf_name) if gguf_name else None
        
        # HF Model selection
        hf_options = [
            "microsoft/phi-2",
            "microsoft/phi-3-mini-4k-instruct",
            "ibm-granite/granite-3.0-8b-instruct",
            "mistralai/Mistral-7B-v0.1",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Custom...",
        ]
        
        default_idx = 0
        if suggested_hf and suggested_hf in hf_options:
            default_idx = hf_options.index(suggested_hf)
        
        selected_hf = st.selectbox(
            "HuggingFace Model for Training",
            hf_options,
            index=default_idx
        )
        
        if selected_hf == "Custom...":
            selected_hf = st.text_input("Enter HuggingFace Model ID", placeholder="org/model-name")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_hf_mapping = False
                st.session_state.selected_gguf = None
                st.rerun()
        
        with col2:
            if st.button("✓ Use This Model", type="primary", use_container_width=True):
                if selected_hf and selected_hf != "Custom...":
                    st.session_state.selected_model = selected_hf
                    st.session_state.model_config = {
                        "name": Path(selected_hf).name,
                        "id": selected_hf,
                        "size": "Unknown",
                        "vram": "~4-6GB",
                        "description": f"Training model for {gguf_name}",
                        "recommended": False,
                        "gguf_path": gguf_path,
                    }
                    st.session_state.show_hf_mapping = False
                    st.success(f"✅ Selected {selected_hf} for training")
                    st.rerun()
                else:
                    st.error("Please select or enter a valid model ID")


def render_recommended_models():
    """Render recommended model cards."""
    st.subheader("Select a Model")
    
    # Filter for recommended
    recommended = [m for m in RECOMMENDED_MODELS if m["recommended"]]
    others = [m for m in RECOMMENDED_MODELS if not m["recommended"]]
    
    st.markdown("#### ⭐ Recommended for 8GB VRAM")
    
    cols = st.columns(3)
    for i, model in enumerate(recommended):
        with cols[i % 3]:
            render_model_card(model)
    
    with st.expander("More Models"):
        cols = st.columns(3)
        for i, model in enumerate(others):
            with cols[i % 3]:
                render_model_card(model)


def render_model_card(model: dict):
    """Render a single model card."""
    is_selected = st.session_state.selected_model == model["id"]
    
    border_color = "#4CAF50" if is_selected else "#ddd"
    bg_color = "#f0fff0" if is_selected else "#fff"
    
    st.markdown(f"""
    <div style="
        padding: 1rem;
        border: 2px solid {border_color};
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: {bg_color};
    ">
        <h4 style="margin: 0;">{model['name']}</h4>
        <p style="color: #666; margin: 0.5rem 0; font-size: 0.85rem;">
            {model['description']}
        </p>
        <p style="margin: 0; font-size: 0.9rem;">
            <strong>Size:</strong> {model['size']} &nbsp;|&nbsp;
            <strong>VRAM:</strong> {model['vram']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    button_label = "✓ Selected" if is_selected else "Select"
    button_type = "secondary" if is_selected else "primary"
    
    if st.button(button_label, key=f"select_{model['id']}", type=button_type, use_container_width=True):
        st.session_state.selected_model = model["id"]
        st.session_state.model_config = model
        st.rerun()


def render_custom_model():
    """Render custom model input."""
    st.subheader("Custom HuggingFace Model")
    
    st.markdown("""
    Enter any HuggingFace model ID compatible with PEFT/LoRA. 
    The model must support `AutoModelForCausalLM`.
    """)
    
    custom_model = st.text_input(
        "HuggingFace Model ID",
        placeholder="e.g., organization/model-name",
        help="Enter the full HuggingFace model path"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        custom_size = st.selectbox(
            "Approximate Size",
            ["1-3B", "3-7B", "7-13B", "13B+"],
            index=1,
            help="Select the approximate model size"
        )
    
    with col2:
        custom_vram = st.selectbox(
            "Estimated VRAM (4-bit)",
            ["~2GB", "~3GB", "~4GB", "~5GB", "~6GB", "~7GB+"],
            index=2,
            help="Estimated VRAM with 4-bit quantization"
        )
    
    if st.button("Use Custom Model", type="primary"):
        if custom_model:
            st.session_state.selected_model = custom_model
            st.session_state.model_config = {
                "name": custom_model.split("/")[-1],
                "id": custom_model,
                "size": custom_size,
                "vram": custom_vram,
                "description": "Custom model",
                "recommended": False,
            }
            st.success(f"✅ Selected: {custom_model}")
            st.rerun()
        else:
            st.error("Please enter a model ID")


def render_selected_model_info():
    """Render information about the selected model."""
    st.subheader("📋 Selected Model")
    
    model_config = st.session_state.model_config
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        **Model:** `{model_config['id']}`
        
        **Size:** {model_config['size']} parameters
        
        **Estimated VRAM:** {model_config['vram']} (with 4-bit quantization)
        
        **Description:** {model_config['description']}
        """)
    
    with col2:
        # VRAM estimation
        loader = ModelLoader()
        
        # Estimate based on size
        size_map = {"1-3B": 2.0, "3-7B": 4.0, "7-13B": 6.0, "13B+": 10.0}
        size_str = model_config.get("size", "3-7B")
        if "B" in size_str:
            # Extract number from "2.7B" or "7B"
            try:
                size_gb = float(size_str.replace("B", "")) * 0.5  # Rough estimate
            except:
                size_gb = 4.0
        else:
            size_gb = size_map.get(size_str, 4.0)
        
        estimate = loader.estimate_training_vram(model_size_gb=size_gb)
        
        st.markdown("**VRAM Breakdown:**")
        st.markdown(f"""
        - Model: ~{estimate['model']:.1f} GB
        - LoRA: ~{estimate['lora_adapters']:.1f} GB
        - Optimizer: ~{estimate['optimizer']:.1f} GB
        - **Total: ~{estimate['total']:.1f} GB**
        """)
        
        if estimate['fits_8gb']:
            st.success("✅ Fits 8GB VRAM")
        else:
            st.warning("⚠️ May not fit 8GB VRAM - reduce seq length")
    
    # Local GGUF models info
    with st.expander("ℹ️ About Local GGUF Models"):
        st.markdown("""
        **Note:** For training, we use HuggingFace format models (not GGUF).
        
        GGUF files are great for inference but cannot be directly fine-tuned.
        After training, you can export your adapter and use it with GGUF models.
        
        **If you have local GGUF models:**
        - Training uses the HuggingFace version of the same model
        - Your LoRA adapter can be used with both formats
        - Export to GGUF coming in Milestone 3
        """)
        
        # Show local GGUF models
        loader = ModelLoader()
        local_models = loader.scan_models()
        
        if local_models:
            st.markdown("**Your local GGUF models:**")
            for model in local_models:
                st.markdown(f"- `{model.name}` ({model.size_gb:.1f} GB)")
        else:
            st.info("No local GGUF models found in `models/base/`")
