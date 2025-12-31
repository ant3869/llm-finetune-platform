"""
Data Preparation Page - Step 1

Upload, preview, and validate training data.
Supports JSON, JSONL, CSV, TXT, PDF, and HTML files.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.dataset_handler import DatasetHandler, TrainingSample


def render_data_prep():
    """Render the data preparation page."""
    st.title("📄 Step 1: Data Preparation")
    st.markdown("Upload your training data or use pre-built IT support templates.")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📤 Upload File", "📋 Use Template", "✏️ Manual Entry"])
    
    with tab1:
        render_file_upload()
    
    with tab2:
        render_templates()
    
    with tab3:
        render_manual_entry()
    
    # Show loaded data preview
    if st.session_state.training_samples:
        st.divider()
        render_data_preview()
        
        # Navigation button
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("Next: Select Model →", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()


def render_file_upload():
    """Render file upload section."""
    st.subheader("Upload Training Data")
    
    # File format info
    with st.expander("ℹ️ Supported Formats", expanded=False):
        st.markdown("""
        **JSON/JSONL (Recommended)**
        ```json
        {
            "instruction": "Your task description",
            "input": "Optional context or input",
            "output": "Expected response"
        }
        ```
        
        **CSV** - Columns: `instruction`, `input`, `output`
        
        **Plain Text** - Q&A pairs or will be auto-chunked
        
        **PDF** - Text will be extracted and chunked
        
        **HTML** - Confluence exports or web pages
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["json", "jsonl", "csv", "txt", "pdf", "html", "htm"],
        help="Upload your training data file"
    )
    
    if uploaded_file is not None:
        # Save to temp location
        temp_path = Path("./data/uploads") / uploaded_file.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process file
        if st.button("📥 Load Data", type="primary"):
            with st.spinner("Processing file..."):
                try:
                    handler = DatasetHandler()
                    samples = handler.load_file(temp_path)
                    
                    if samples:
                        st.session_state.training_samples = samples
                        st.session_state.dataset_stats = handler.get_statistics(samples)
                        st.success(f"✅ Loaded {len(samples)} training samples!")
                        st.rerun()
                    else:
                        st.error("No valid samples found in file.")
                        
                except Exception as e:
                    st.error(f"Failed to load file: {e}")


def render_templates():
    """Render template selection section."""
    st.subheader("IT Support Templates")
    st.markdown("Start with pre-built templates for common IT support scenarios.")
    
    templates_dir = Path("./data/templates")
    
    if not templates_dir.exists():
        st.warning("Templates directory not found. Create `data/templates/` with sample files.")
        return
    
    # Find template files
    template_files = list(templates_dir.glob("*.json"))
    
    if not template_files:
        st.info("No template files found in `data/templates/`")
        return
    
    # Display templates as cards
    cols = st.columns(3)
    
    template_info = {
        "servicenow_ticket": ("🎫", "ServiceNow Tickets", "Ticket analysis and resolution"),
        "knowledge_article": ("📚", "Knowledge Articles", "KB-based Q&A"),
        "sop_format": ("📋", "SOPs & Procedures", "Step-by-step guides"),
    }
    
    for i, template_file in enumerate(template_files):
        template_name = template_file.stem
        icon, title, desc = template_info.get(
            template_name, 
            ("📄", template_name.replace("_", " ").title(), "Custom template")
        )
        
        with cols[i % 3]:
            st.markdown(f"""
            <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 1rem;">
                <h3>{icon} {title}</h3>
                <p style="color: #666; font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Load {title}", key=f"load_{template_name}"):
                with st.spinner(f"Loading {title}..."):
                    try:
                        handler = DatasetHandler()
                        samples = handler.load_file(template_file)
                        
                        st.session_state.training_samples = samples
                        st.session_state.dataset_stats = handler.get_statistics(samples)
                        st.success(f"✅ Loaded {len(samples)} samples from template!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Failed to load template: {e}")


def render_manual_entry():
    """Render manual data entry section."""
    st.subheader("Manual Entry")
    st.markdown("Add training samples manually.")
    
    # Form for new sample
    with st.form("manual_entry_form"):
        instruction = st.text_area(
            "Instruction",
            placeholder="What task should the model perform?",
            help="The instruction or question for the model"
        )
        
        input_text = st.text_area(
            "Input (Optional)",
            placeholder="Additional context or input data",
            help="Optional context the model should use"
        )
        
        output = st.text_area(
            "Expected Output",
            placeholder="The ideal response from the model",
            help="The response you want the model to learn"
        )
        
        submitted = st.form_submit_button("➕ Add Sample", type="primary")
        
        if submitted:
            if instruction and output:
                sample = TrainingSample(
                    instruction=instruction,
                    input=input_text,
                    output=output
                )
                
                if st.session_state.training_samples is None:
                    st.session_state.training_samples = []
                
                st.session_state.training_samples.append(sample)
                
                # Update stats
                handler = DatasetHandler()
                st.session_state.dataset_stats = handler.get_statistics(
                    st.session_state.training_samples
                )
                
                st.success("✅ Sample added!")
                st.rerun()
            else:
                st.error("Instruction and Output are required.")
    
    # Show current manual samples count
    if st.session_state.training_samples:
        st.info(f"Current samples: {len(st.session_state.training_samples)}")
        
        if st.button("🗑️ Clear All Samples"):
            st.session_state.training_samples = None
            st.session_state.dataset_stats = None
            st.rerun()


def render_data_preview():
    """Render data preview and statistics."""
    st.subheader("📊 Dataset Preview")
    
    samples = st.session_state.training_samples
    stats = st.session_state.dataset_stats
    
    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", stats.total_samples)
    
    with col2:
        st.metric("Training", stats.train_samples)
    
    with col3:
        st.metric("Validation", stats.validation_samples)
    
    with col4:
        st.metric("Avg Output Length", f"{stats.avg_output_length:.0f} words")
    
    # Warnings
    if stats.warnings:
        for warning in stats.warnings:
            st.warning(f"⚠️ {warning}")
    
    # Sample preview
    st.markdown("### Sample Preview")
    
    # Convert to displayable format
    preview_data = []
    for i, sample in enumerate(samples[:10]):  # Show first 10
        preview_data.append({
            "#": i + 1,
            "Instruction": sample.instruction[:100] + "..." if len(sample.instruction) > 100 else sample.instruction,
            "Input": sample.input[:50] + "..." if len(sample.input) > 50 else sample.input,
            "Output": sample.output[:100] + "..." if len(sample.output) > 100 else sample.output,
        })
    
    df = pd.DataFrame(preview_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    if len(samples) > 10:
        st.caption(f"Showing 10 of {len(samples)} samples")
    
    # Detailed view of single sample
    with st.expander("🔍 View Full Sample"):
        sample_idx = st.selectbox(
            "Select sample",
            range(len(samples)),
            format_func=lambda x: f"Sample {x + 1}: {samples[x].instruction[:50]}..."
        )
        
        selected = samples[sample_idx]
        
        st.markdown("**Instruction:**")
        st.text(selected.instruction)
        
        if selected.input:
            st.markdown("**Input:**")
            st.text(selected.input)
        
        st.markdown("**Output:**")
        st.text(selected.output)
        
        st.markdown("**Formatted Prompt:**")
        st.code(selected.format_prompt("alpaca"), language="text")
