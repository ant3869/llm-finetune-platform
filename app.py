"""
LLM Fine-Tuning Platform - Main Streamlit Application

A user-friendly interface for fine-tuning local LLMs on consumer hardware.
Optimized for 8GB VRAM GPUs using QLoRA.

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="LLM Fine-Tuning Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Card-like containers */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px;
    }
    
    /* Success/error message styling */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        # Data state
        "uploaded_file": None,
        "training_samples": None,
        "dataset_stats": None,
        
        # Model state
        "selected_model": None,
        "model_config": None,
        
        # Training state
        "training_config": None,
        "training_progress": None,
        "is_training": False,
        "trainer": None,
        "training_error": None,  # Store training errors to persist across reruns
        
        # Navigation
        "current_step": 1,
        
        # Results
        "trained_adapter_path": None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def render_sidebar():
    """Render the sidebar with navigation and status."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.title("LLM Fine-Tuning")
        st.caption("v0.4.0 - Milestone 4")
        
        st.divider()
        
        # Navigation steps
        st.subheader("📋 Workflow")
        
        steps = [
            ("1️⃣", "Data Preparation", 1),
            ("2️⃣", "Model Selection", 2),
            ("3️⃣", "Training", 3),
            ("4️⃣", "Evaluation", 4),
            ("5️⃣", "Export", 5),
        ]
        
        for icon, name, step_num in steps:
            # Determine step status
            if step_num < st.session_state.current_step:
                status = "✅"
            elif step_num == st.session_state.current_step:
                status = "👉"
            else:
                status = "⬜"
            
            # Check if step is accessible
            is_accessible = step_num <= st.session_state.current_step + 1
            
            if is_accessible:  # All 5 steps now implemented
                if st.button(f"{status} {icon} {name}", key=f"nav_{step_num}", use_container_width=True):
                    st.session_state.current_step = step_num
                    st.rerun()
            else:
                st.button(f"{status} {icon} {name}", key=f"nav_{step_num}", use_container_width=True, disabled=True)
        
        st.divider()
        
        # Status summary
        st.subheader("📊 Status")
        
        # Data status
        if st.session_state.training_samples:
            st.success(f"📄 {len(st.session_state.training_samples)} samples loaded")
        else:
            st.info("📄 No data loaded")
        
        # Model status
        if st.session_state.selected_model:
            st.success(f"🤖 Model: {st.session_state.selected_model[:20]}...")
        else:
            st.info("🤖 No model selected")
        
        # Training status
        if st.session_state.is_training:
            st.warning("🔄 Training in progress...")
        elif st.session_state.trained_adapter_path:
            st.success("✅ Training complete!")
        
        st.divider()
        
        # GPU Status
        st.subheader("💾 GPU Status")
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    used_mem = torch.cuda.memory_allocated(0) / (1024**3)
                    
                    st.caption(f"**{gpu_name}**")
                    st.progress(used_mem / total_mem if total_mem > 0 else 0)
                    st.caption(f"{used_mem:.1f} / {total_mem:.1f} GB")
                else:
                    st.warning("CUDA available but no devices")
            else:
                st.warning("No CUDA GPU detected")
                st.caption("Check PyTorch CUDA install")
                # Show diagnostic info
                with st.expander("Diagnostics"):
                    st.code(f"torch.cuda.is_available(): {cuda_available}")
                    st.code(f"torch version: {torch.__version__}")
                    if hasattr(torch.version, 'cuda'):
                        st.code(f"CUDA version: {torch.version.cuda}")
        except ImportError:
            st.info("PyTorch not installed")
        except Exception as e:
            st.error(f"GPU check failed: {e}")


def render_main_content():
    """Render the main content based on current step."""
    step = st.session_state.current_step
    
    if step == 1:
        render_data_prep_page()
    elif step == 2:
        render_model_select_page()
    elif step == 3:
        render_training_page()
    elif step == 4:
        render_evaluation_page()
    elif step == 5:
        render_export_page()


def render_data_prep_page():
    """Render data preparation page - imported from ui/pages."""
    try:
        from ui.pages.data_prep import render_data_prep
        render_data_prep()
    except ImportError as e:
        st.error(f"Failed to load data prep page: {e}")
        st.info("Make sure ui/pages/data_prep.py exists")


def render_model_select_page():
    """Render model selection page - imported from ui/pages."""
    try:
        from ui.pages.model_select import render_model_select
        render_model_select()
    except ImportError as e:
        st.error(f"Failed to load model select page: {e}")
        st.info("Make sure ui/pages/model_select.py exists")


def render_training_page():
    """Render training page - imported from ui/pages."""
    try:
        from ui.pages.training import render_training
        render_training()
    except ImportError as e:
        st.error(f"Failed to load training page: {e}")
        st.info("Make sure ui/pages/training.py exists")


def render_evaluation_page():
    """Render evaluation/testing page - imported from ui/pages."""
    try:
        from ui.pages.evaluation import render_evaluation
        render_evaluation()
    except ImportError as e:
        st.error(f"Failed to load evaluation page: {e}")
        st.info("Make sure ui/pages/evaluation.py exists")


def render_export_page():
    """Render export page - imported from ui/pages."""
    try:
        from ui.pages.export import render_export
        render_export()
    except ImportError as e:
        st.error(f"Failed to load export page: {e}")
        st.info("Make sure ui/pages/export.py exists")


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
