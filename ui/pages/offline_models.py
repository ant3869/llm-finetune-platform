"""
Offline Models Page - Model Download and Management.

Provides UI for:
- Viewing available offline model downloads
- Generating download instructions for air-gapped environments  
- Managing local model cache
- Setting up offline environment
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_offline_models():
    """Render the offline models management page."""
    st.title("📦 Offline Model Manager")
    st.markdown("Download and manage models for air-gapped or restricted network environments.")
    
    # Initialize offline manager
    try:
        from core.offline_models import OfflineModelManager, get_offline_env_setup
        manager = OfflineModelManager()
    except ImportError as e:
        st.error(f"Failed to load offline manager: {e}")
        return
    
    # Tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔽 Download Models",
        "📂 Local Models", 
        "📖 Offline Guide",
        "⚙️ Environment Setup"
    ])
    
    with tab1:
        render_download_section(manager)
    
    with tab2:
        render_local_models_section(manager)
    
    with tab3:
        render_offline_guide(manager)
    
    with tab4:
        render_env_setup()


def render_download_section(manager):
    """Render model download section."""
    st.subheader("Available Models for Download")
    st.markdown("These models can be downloaded for offline use. GGUF models are recommended for inference.")
    
    models = manager.get_available_offline_models()
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        format_filter = st.selectbox("Filter by format", ["All", "GGUF", "HuggingFace"])
    with col2:
        use_filter = st.selectbox("Filter by use case", ["All", "Inference", "Training"])
    
    st.divider()
    
    for model in models:
        # Apply filters
        if format_filter != "All" and model["format"].lower() != format_filter.lower():
            continue
        if use_filter != "All" and use_filter.lower() not in [r.lower() for r in model["recommended_for"]]:
            continue
        
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                status_icon = "✅" if model["is_downloaded"] else "⬇️"
                st.markdown(f"### {status_icon} {model['name']}")
                st.caption(model["description"])
                st.caption(f"Format: {model['format'].upper()} | Size: ~{model['file_size_gb']:.1f} GB")
            
            with col2:
                use_badges = " ".join([f"`{r}`" for r in model["recommended_for"]])
                st.markdown(f"**Use:** {use_badges}")
            
            with col3:
                if model["is_downloaded"]:
                    st.success("Downloaded")
                else:
                    if st.button("📋 Instructions", key=f"inst_{model['key']}"):
                        st.session_state[f"show_instructions_{model['key']}"] = True
                    
                    # Try direct download button
                    if model.get("download_url"):
                        if st.button("⬇️ Download", key=f"dl_{model['key']}", type="primary"):
                            with st.spinner(f"Downloading {model['name']}..."):
                                success, msg = manager.download_model(model["key"])
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
            
            # Show instructions if requested
            if st.session_state.get(f"show_instructions_{model['key']}", False):
                with st.expander("📋 Download Instructions", expanded=True):
                    instructions = manager.generate_download_instructions(model["key"])
                    st.code(instructions, language="text")
                    
                    if st.button("Close", key=f"close_{model['key']}"):
                        st.session_state[f"show_instructions_{model['key']}"] = False
                        st.rerun()
            
            st.divider()


def render_local_models_section(manager):
    """Render local models browser."""
    st.subheader("Local Models")
    st.markdown("Models currently available on this machine.")
    
    if st.button("🔄 Refresh", key="refresh_local"):
        st.rerun()
    
    local_models = manager.scan_local_models()
    
    # GGUF Models
    st.markdown("### GGUF Models (Inference)")
    if local_models["gguf"]:
        for model in local_models["gguf"]:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{model['name']}**")
                st.caption(f"Path: {model['path']}")
            with col2:
                st.caption(f"{model['size_gb']:.2f} GB")
            with col3:
                if st.button("Use", key=f"use_gguf_{model['name']}"):
                    st.session_state.selected_gguf_model = model["path"]
                    st.success(f"Selected: {model['name']}")
    else:
        st.info("No GGUF models found. Place .gguf files in `./models/base/`")
    
    st.divider()
    
    # HuggingFace Models
    st.markdown("### HuggingFace Models (Training)")
    if local_models["huggingface"]:
        for model in local_models["huggingface"]:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{model['name']}**")
                st.caption(f"Path: {model['path']}")
            with col2:
                st.caption(f"{model['size_gb']:.2f} GB")
            with col3:
                # Verify model files
                is_valid, missing = manager.verify_model_files(Path(model["path"]))
                if is_valid:
                    st.success("✅ Valid")
                else:
                    st.warning(f"⚠️ Missing: {', '.join(missing[:2])}")
    else:
        st.info("No HuggingFace models cached locally.")
        st.markdown("""
        To cache a model for offline use:
        1. Download on a machine with internet
        2. Copy to `./models/cache/huggingface/`
        """)


def render_offline_guide(manager):
    """Render complete offline workflow guide."""
    st.subheader("📖 Complete Offline Workflow Guide")
    
    guide = manager.get_offline_workflow_guide()
    st.code(guide, language="text")
    
    # Quick links
    st.markdown("### Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**For Inference (GGUF):**")
        if st.button("📋 Phi-2 GGUF Instructions"):
            st.code(manager.generate_download_instructions("phi-2-gguf"), language="text")
        
        if st.button("📋 Mistral 7B GGUF Instructions"):
            st.code(manager.generate_download_instructions("mistral-7b-gguf"), language="text")
    
    with col2:
        st.markdown("**For Training (HuggingFace):**")
        if st.button("📋 microsoft/phi-2 Instructions"):
            st.code(manager.generate_hf_offline_instructions("microsoft/phi-2"), language="text")
        
        if st.button("📋 ibm-granite Instructions"):
            st.code(manager.generate_hf_offline_instructions("ibm-granite/granite-3.0-8b-instruct"), language="text")


def render_env_setup():
    """Render environment setup section."""
    st.subheader("⚙️ Offline Environment Setup")
    st.markdown("Configure your environment for fully offline operation.")
    
    from core.offline_models import get_offline_env_setup
    
    # Environment variables
    st.markdown("### Environment Variables")
    st.markdown("Set these variables to prevent HuggingFace from trying to connect online:")
    
    st.code(get_offline_env_setup(), language="bash")
    
    # Quick setup script
    st.markdown("### Quick Setup Script")
    
    tab1, tab2 = st.tabs(["Windows (PowerShell)", "Linux/Mac (Bash)"])
    
    with tab1:
        ps_script = """# Save as setup_offline.ps1 and run before starting the platform
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1" 
$env:HF_HOME = "$PWD\\models\\cache\\huggingface"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

Write-Host "Offline mode enabled!" -ForegroundColor Green
Write-Host "HF_HOME set to: $env:HF_HOME"

# Now start the platform
# streamlit run app.py
"""
        st.code(ps_script, language="powershell")
        
        if st.button("📋 Copy PowerShell Script"):
            st.write("Script copied! (Use Ctrl+C on the code block)")
    
    with tab2:
        bash_script = """#!/bin/bash
# Save as setup_offline.sh and source before starting the platform
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=./models/cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1

echo "Offline mode enabled!"
echo "HF_HOME set to: $HF_HOME"

# Now start the platform
# streamlit run app.py
"""
        st.code(bash_script, language="bash")
    
    # Current environment status
    st.markdown("### Current Environment Status")
    
    import os
    env_vars = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "Not set"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "Not set"),
        "HF_HOME": os.environ.get("HF_HOME", "Default (~/.cache/huggingface)"),
        "HF_HUB_DISABLE_TELEMETRY": os.environ.get("HF_HUB_DISABLE_TELEMETRY", "Not set"),
    }
    
    for var, value in env_vars.items():
        is_offline = value == "1" if var != "HF_HOME" else value != "Default (~/.cache/huggingface)"
        icon = "✅" if is_offline else "⚠️"
        st.markdown(f"{icon} **{var}**: `{value}`")
    
    offline_ready = env_vars["HF_HUB_OFFLINE"] == "1" and env_vars["TRANSFORMERS_OFFLINE"] == "1"
    
    if offline_ready:
        st.success("✅ Environment is configured for offline mode!")
    else:
        st.warning("⚠️ Environment is NOT fully configured for offline mode. Set the variables above.")
