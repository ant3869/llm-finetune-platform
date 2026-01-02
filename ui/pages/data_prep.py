"""
Data Preparation Page - Step 1

Upload, preview, and validate training data.
Supports JSON, JSONL, CSV, TXT, PDF, and HTML files.
Includes advanced data cleaning options.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.dataset_handler import DatasetHandler, TrainingSample
from ui.components.hf_datasets_browser import render_hf_datasets_browser


def render_data_prep():
    """Render the data preparation page."""
    st.title("📄 Step 1: Data Preparation")
    st.markdown("Upload your training data or use pre-built IT support templates.")
    
    # Create tabs for different input methods
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload File", 
        "📋 Use Template", 
        "🤗 HuggingFace", 
        "✏️ Manual Entry", 
        "🧹 Data Cleaning"
    ])
    
    with tab1:
        render_file_upload()
    
    with tab2:
        render_templates()
    
    with tab3:
        render_hf_datasets_browser()
    
    with tab4:
        render_manual_entry()
    
    with tab5:
        render_data_cleaning()
    
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
    st.subheader("📋 Training Templates")
    st.markdown("Start with pre-built templates for common training scenarios.")
    
    templates_dir = Path("./data/templates")
    
    if not templates_dir.exists():
        st.warning("Templates directory not found. Create `data/templates/` with sample files.")
        return
    
    # Find template files
    template_files = list(templates_dir.glob("*.json"))
    
    if not template_files:
        st.info("No template files found in `data/templates/`")
        return
    
    # Template categories for grouping
    template_info = {
        # IT Support templates
        "servicenow_ticket": ("🎫", "ServiceNow Tickets", "Ticket analysis and resolution", "IT Support"),
        "knowledge_article": ("📚", "Knowledge Articles", "KB-based Q&A", "IT Support"),
        "sop_format": ("📋", "SOPs & Procedures", "Step-by-step guides", "IT Support"),
        "ticket_triage": ("🏷️", "Ticket Triage", "Priority & category assignment", "IT Support"),
        "incident_postmortem": ("📊", "Incident Postmortems", "Root cause analysis reports", "IT Support"),
        "change_request": ("🔄", "Change Requests", "Change management & approvals", "IT Support"),
        "customer_communication": ("💬", "Customer Communication", "Service incident updates", "IT Support"),
        "runbook": ("📖", "Runbooks", "Alert handling procedures", "IT Support"),
        # Book & PDF templates
        "book_chapter_summary": ("📖", "Chapter Summaries", "Summarize book chapters concisely", "Books & PDFs"),
        "book_qa": ("❓", "Book Q&A", "Question answering from book content", "Books & PDFs"),
        "book_knowledge_extraction": ("🧠", "Knowledge Extraction", "Extract structured knowledge from books", "Books & PDFs"),
        "pdf_text_cleaning": ("🧹", "PDF Text Cleaning", "Clean and format raw PDF text", "Books & PDFs"),
    }
    
    # Group templates by category
    categorized = {}
    for template_file in template_files:
        template_name = template_file.stem
        info = template_info.get(
            template_name, 
            ("📄", template_name.replace("_", " ").title(), "Custom template", "Other")
        )
        if len(info) == 3:
            icon, title, desc = info
            category = "Other"
        else:
            icon, title, desc, category = info
        
        if category not in categorized:
            categorized[category] = []
        categorized[category].append((template_file, icon, title, desc))
    
    # Display templates grouped by category
    for category, templates in categorized.items():
        st.markdown(f"### {category}")
        cols = st.columns(3)
        
        for i, (template_file, icon, title, desc) in enumerate(templates):
            template_name = template_file.stem
            with cols[i % 3]:
                st.markdown(f"""
                <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 1rem;">
                    <h4>{icon} {title}</h4>
                    <p style="color: #666; font-size: 0.85rem; margin: 0.5rem 0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Load {title}", key=f"load_{template_name}", use_container_width=True):
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
    
    # If stats is None, recalculate it
    if stats is None and samples:
        handler = DatasetHandler()
        stats = handler.get_statistics(samples)
        st.session_state.dataset_stats = stats
    
    # Statistics cards (only show if stats available)
    if stats:
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


def render_data_cleaning():
    """Render the data cleaning options tab."""
    st.subheader("🧹 Data Cleaning")
    st.markdown("Clean and preprocess your loaded training data for better quality.")
    
    if not st.session_state.training_samples:
        st.info("📤 Load data first using the Upload or Template tabs, then return here to clean it.")
        return
    
    st.success(f"✅ {len(st.session_state.training_samples)} samples loaded and ready for cleaning")
    
    # Cleaning options in expandable sections
    with st.expander("🔧 Quick Cleaning Presets", expanded=True):
        preset = st.radio(
            "Select a cleaning preset",
            ["None", "Minimal", "Standard", "Aggressive", "IT Support"],
            horizontal=True,
            help="Presets apply common cleaning operations"
        )
        
        preset_descriptions = {
            "None": "No automatic cleaning",
            "Minimal": "Unicode normalization, whitespace cleanup",
            "Standard": "Minimal + HTML removal, quote normalization",
            "Aggressive": "Standard + URL/email removal, deduplication",
            "IT Support": "Optimized for ServiceNow/IT ticket data",
        }
        st.caption(preset_descriptions.get(preset, ""))
    
    with st.expander("✂️ Delimiter/Prefix Removal", expanded=False):
        st.markdown("""
        Remove text before or after a delimiter on each line.  
        Example: `"XX | Actual text"` → `"Actual text"` with delimiter `"XX | "`
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            prefix_delimiter = st.text_input(
                "Prefix delimiter to remove",
                placeholder="e.g., XX | or >>",
                help="Everything before and including this will be removed from each line"
            )
        with col2:
            suffix_delimiter = st.text_input(
                "Suffix delimiter to remove", 
                placeholder="e.g., | END or ##",
                help="Everything after and including this will be removed from each line"
            )
    
    with st.expander("🔍 Pattern Removal (Regex)", expanded=False):
        st.markdown("Remove text matching regex patterns. One pattern per line.")
        patterns_text = st.text_area(
            "Patterns to remove",
            placeholder="\\d{4}-\\d{2}-\\d{2}\nINC\\d+\n\\[.*?\\]",
            help="Enter regex patterns, one per line"
        )
        
        st.markdown("**Common patterns:**")
        col1, col2 = st.columns(2)
        with col1:
            st.code("\\d{4}-\\d{2}-\\d{2}  # Dates", language="text")
            st.code("INC\\d+  # Incident numbers", language="text")
        with col2:
            st.code("\\[.*?\\]  # Bracketed text", language="text")
            st.code("^\\s*#.*$  # Comment lines", language="text")
    
    with st.expander("⚙️ Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            normalize_unicode = st.checkbox("Normalize Unicode", value=True)
            normalize_whitespace = st.checkbox("Normalize Whitespace", value=True)
            remove_html = st.checkbox("Remove HTML Tags", value=True)
            remove_markdown = st.checkbox("Remove Markdown", value=False)
        
        with col2:
            remove_urls = st.checkbox("Remove URLs", value=False)
            remove_emails = st.checkbox("Remove Emails", value=False)
            remove_duplicates = st.checkbox("Remove Duplicates", value=True)
            lowercase = st.checkbox("Convert to Lowercase", value=False)
        
        st.markdown("**Quality Filters:**")
        col1, col2 = st.columns(2)
        with col1:
            min_length = st.number_input("Min output length (chars)", value=10, min_value=0)
        with col2:
            min_words = st.number_input("Min word count", value=3, min_value=0)
    
    # Preview and Apply buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        preview_btn = st.button("👁️ Preview Changes", use_container_width=True)
    
    with col2:
        apply_btn = st.button("✅ Apply Cleaning", type="primary", use_container_width=True)
    
    # Handle preview/apply
    if preview_btn or apply_btn:
        try:
            from core.data_cleaner import CleaningConfig, DataCleaningPipeline
            
            # Build config from UI options
            patterns_to_remove = []
            if patterns_text:
                patterns_to_remove = [p.strip() for p in patterns_text.split('\n') if p.strip()]
            
            # Map preset to config
            preset_map = {
                "None": None,
                "Minimal": "minimal",
                "Standard": "standard", 
                "Aggressive": "aggressive",
                "IT Support": "it_support",
            }
            
            if preset != "None":
                pipeline = DataCleaningPipeline(preset=preset_map[preset])
                config = pipeline.config
            else:
                config = CleaningConfig()
            
            # Apply custom overrides
            if prefix_delimiter:
                config.prefix_delimiter = prefix_delimiter
            if suffix_delimiter:
                config.suffix_delimiter = suffix_delimiter
            if patterns_to_remove:
                config.patterns_to_remove = patterns_to_remove
            
            # Advanced options (only if not using preset or want to override)
            config.normalize_unicode = normalize_unicode
            config.normalize_whitespace = normalize_whitespace
            config.remove_html_tags = remove_html
            config.remove_markdown = remove_markdown
            config.remove_urls = remove_urls
            config.remove_emails = remove_emails
            config.remove_duplicates = remove_duplicates
            config.lowercase = lowercase
            config.min_length = min_length
            config.min_word_count = min_words
            
            # Create pipeline and clean
            pipeline = DataCleaningPipeline(config=config)
            cleaned_samples = pipeline.clean_samples(st.session_state.training_samples)
            stats = pipeline.get_stats()
            
            # Show results
            st.markdown("### Cleaning Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original", stats["processed"])
            with col2:
                st.metric("Passed", stats["passed"])
            with col3:
                st.metric("Filtered", stats["filtered"])
            with col4:
                st.metric("Pass Rate", f"{stats['pass_rate']:.1%}")
            
            if stats["filter_reasons"]:
                with st.expander("📊 Filter Reasons"):
                    for reason, count in stats["filter_reasons"].items():
                        st.write(f"• {reason}: {count}")
            
            # Preview comparison
            if cleaned_samples and preview_btn:
                st.markdown("### Before/After Comparison")
                
                orig = st.session_state.training_samples[0]
                cleaned = cleaned_samples[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original:**")
                    st.text(orig.output[:300])
                with col2:
                    st.markdown("**Cleaned:**")
                    st.text(cleaned.output[:300])
            
            # Apply changes
            if apply_btn and cleaned_samples:
                st.session_state.training_samples = cleaned_samples
                handler = DatasetHandler()
                st.session_state.dataset_stats = handler.get_statistics(cleaned_samples)
                st.success(f"✅ Applied cleaning! {len(cleaned_samples)} samples remaining.")
                st.rerun()
            elif apply_btn and not cleaned_samples:
                st.error("All samples were filtered out! Adjust your cleaning settings.")
                
        except ImportError as e:
            st.error(f"Missing cleaning module: {e}")
        except Exception as e:
            st.error(f"Cleaning failed: {e}")
            import traceback
            st.code(traceback.format_exc())

