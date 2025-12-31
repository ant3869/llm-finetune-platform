"""
Export Page - Step 5

Export fine-tuned models to various formats including GGUF.
"""

import streamlit as st
from pathlib import Path
import sys
import subprocess
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_export():
    """Render the export page."""
    st.title("📦 Step 5: Export")
    st.markdown("Export your fine-tuned model to various formats.")
    
    # Check for trained adapters
    adapter_dir = Path("./models/adapters")
    adapters = []
    if adapter_dir.exists():
        adapters = [d.name for d in adapter_dir.iterdir() 
                   if d.is_dir() and (d / "adapter_config.json").exists()]
    
    if not adapters:
        st.warning("⚠️ No trained adapters found. Complete training first!")
        if st.button("← Go to Training"):
            st.session_state.current_step = 3
            st.rerun()
        return
    
    # Adapter selection
    st.subheader("📁 Select Adapter to Export")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_adapter = st.selectbox(
            "Trained Adapter",
            adapters,
            help="Select the adapter you want to export"
        )
        adapter_path = adapter_dir / selected_adapter
    
    with col2:
        # Show adapter info
        adapter_config = adapter_path / "adapter_config.json"
        if adapter_config.exists():
            import json
            with open(adapter_config) as f:
                config = json.load(f)
            st.metric("Base Model", config.get("base_model_name_or_path", "Unknown")[:30] + "...")
    
    st.divider()
    
    # Export options
    st.subheader("🔧 Export Options")
    
    tab1, tab2, tab3 = st.tabs(["🔀 Merge & Save", "📄 GGUF Conversion", "☁️ Upload to Hub"])
    
    with tab1:
        render_merge_export(adapter_path, selected_adapter)
    
    with tab2:
        render_gguf_export(adapter_path, selected_adapter)
    
    with tab3:
        render_hub_upload(adapter_path, selected_adapter)


def render_merge_export(adapter_path: Path, adapter_name: str):
    """Render merge and save options."""
    st.markdown("""
    ### Merge LoRA into Base Model
    
    Merges your LoRA adapter weights directly into the base model, creating a 
    standalone model that doesn't require PEFT to load.
    
    **Use this when:**
    - You want a single model file without adapter dependencies
    - Deploying to systems without PEFT installed
    - Sharing the complete fine-tuned model
    """)
    
    # Options
    col1, col2 = st.columns(2)
    
    with col1:
        base_model = st.text_input(
            "Base Model ID",
            value=st.session_state.get("selected_model", "microsoft/phi-2"),
            help="The original model used for training"
        )
    
    with col2:
        output_name = st.text_input(
            "Output Name",
            value=f"{adapter_name}_merged",
            help="Name for the merged model"
        )
    
    save_16bit = st.checkbox("Save in 16-bit (larger but more compatible)", value=True)
    
    output_path = Path("./models/merged") / output_name
    
    if st.button("🔀 Merge and Save", type="primary", use_container_width=True):
        with st.spinner("Merging model... This may take several minutes."):
            try:
                merge_and_save(base_model, str(adapter_path), str(output_path), save_16bit)
                st.success(f"✅ Model merged and saved to: `{output_path}`")
            except Exception as e:
                st.error(f"Merge failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())


def merge_and_save(base_model: str, adapter_path: str, output_path: str, save_16bit: bool = True):
    """Merge LoRA adapter into base model and save."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Load and merge adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    # Save
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_16bit:
        model.save_pretrained(output_dir, safe_serialization=True)
    else:
        model.half().save_pretrained(output_dir, safe_serialization=True)
    
    tokenizer.save_pretrained(output_dir)
    
    return output_dir


def render_gguf_export(adapter_path: Path, adapter_name: str):
    """Render GGUF export options."""
    st.markdown("""
    ### Export to GGUF Format
    
    GGUF is a quantized format optimized for inference with llama.cpp. 
    This creates a compact, fast model for deployment.
    
    **Requirements:**
    - First merge the adapter into base model
    - llama.cpp's `convert.py` script
    - Python llama-cpp-python package
    
    **Note:** GGUF conversion requires the merged model, not just the adapter.
    """)
    
    # Check for merged models
    merged_dir = Path("./models/merged")
    merged_models = []
    if merged_dir.exists():
        merged_models = [d.name for d in merged_dir.iterdir() if d.is_dir()]
    
    if not merged_models:
        st.warning("⚠️ No merged models found. Merge your adapter first (see 'Merge & Save' tab).")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_merged = st.selectbox(
            "Select Merged Model",
            merged_models,
            help="Choose the merged model to convert"
        )
        merged_path = merged_dir / selected_merged
    
    with col2:
        quantization = st.selectbox(
            "Quantization",
            ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "F16"],
            index=0,
            help="Quantization method (Q4_K_M is recommended for balance of size/quality)"
        )
    
    st.info("""
    **Quantization Guide:**
    - **Q4_K_M** - 4-bit, good balance (recommended)
    - **Q5_K_M** - 5-bit, better quality, larger
    - **Q8_0** - 8-bit, near-original quality
    - **F16** - 16-bit, no quantization
    """)
    
    if st.button("📄 Convert to GGUF", type="primary", use_container_width=True):
        st.warning("""
        ⚠️ **Manual Conversion Required**
        
        GGUF conversion requires llama.cpp tools. Follow these steps:
        
        1. Clone llama.cpp:
        ```bash
        git clone https://github.com/ggerganov/llama.cpp
        cd llama.cpp
        ```
        
        2. Convert to GGUF:
        ```bash
        python convert_hf_to_gguf.py {merged_path} --outfile {selected_merged}.gguf
        ```
        
        3. Quantize (optional):
        ```bash
        ./llama-quantize {selected_merged}.gguf {selected_merged}-{quantization}.gguf {quantization}
        ```
        """.format(merged_path=merged_path, selected_merged=selected_merged, quantization=quantization))
        
        # Provide copy-paste commands
        st.code(f"""
# Clone llama.cpp (if not already)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# Convert to GGUF
python convert_hf_to_gguf.py "{merged_path}" --outfile "{selected_merged}.gguf"

# Quantize
./llama-quantize "{selected_merged}.gguf" "{selected_merged}-{quantization}.gguf" {quantization}
        """, language="bash")


def render_hub_upload(adapter_path: Path, adapter_name: str):
    """Render HuggingFace Hub upload options."""
    st.markdown("""
    ### Upload to HuggingFace Hub
    
    Share your fine-tuned adapter on the HuggingFace Hub for easy access
    and collaboration.
    
    **Requirements:**
    - HuggingFace account
    - `huggingface_hub` package
    - HF token with write access
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        repo_name = st.text_input(
            "Repository Name",
            value=adapter_name,
            help="Name for the HuggingFace repository"
        )
    
    with col2:
        private = st.checkbox("Private Repository", value=True)
    
    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        help="Token with write access from huggingface.co/settings/tokens"
    )
    
    model_card = st.text_area(
        "Model Card (README.md)",
        value=f"""---
tags:
- fine-tuned
- lora
- qlora
license: apache-2.0
---

# {repo_name}

Fine-tuned LoRA adapter created with LLM Fine-Tuning Platform.

## Training Details

- Base model: {st.session_state.get("selected_model", "Unknown")}
- Training method: QLoRA (4-bit quantization)
- Created: {Path(adapter_path).stat().st_mtime if adapter_path.exists() else "Unknown"}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("BASE_MODEL_ID")
model = PeftModel.from_pretrained(base_model, "{repo_name}")
```
""",
        height=300,
        help="README content for your model"
    )
    
    if st.button("☁️ Upload to Hub", type="primary", use_container_width=True):
        if not hf_token:
            st.error("Please provide your HuggingFace token")
            return
        
        with st.spinner("Uploading to HuggingFace Hub..."):
            try:
                upload_to_hub(str(adapter_path), repo_name, hf_token, private, model_card)
                st.success(f"✅ Uploaded to: https://huggingface.co/{repo_name}")
            except Exception as e:
                st.error(f"Upload failed: {e}")


def upload_to_hub(adapter_path: str, repo_name: str, token: str, private: bool, model_card: str):
    """Upload adapter to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi(token=token)
    
    # Create repo
    try:
        create_repo(repo_name, private=private, token=token, exist_ok=True)
    except Exception:
        pass  # Repo might already exist
    
    # Write model card
    adapter_dir = Path(adapter_path)
    readme_path = adapter_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    
    # Upload
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_name,
        token=token,
    )
