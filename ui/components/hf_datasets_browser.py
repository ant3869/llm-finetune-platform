"""
HuggingFace Datasets Browser

Browse, search, and load datasets from HuggingFace Hub for fine-tuning.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


# Popular fine-tuning datasets categorized
CURATED_DATASETS = {
    "Instruction Following": [
        {
            "id": "databricks/databricks-dolly-15k",
            "name": "Databricks Dolly 15k",
            "description": "15k instruction-following records for training language models",
            "size": "~15,000 samples",
            "license": "CC BY-SA 3.0",
            "use_case": "General instruction following, Q&A, summarization"
        },
        {
            "id": "tatsu-lab/alpaca",
            "name": "Stanford Alpaca",
            "description": "52k instructions generated using self-instruct from GPT",
            "size": "~52,000 samples",
            "license": "CC BY-NC 4.0",
            "use_case": "General instruction following"
        },
        {
            "id": "OpenAssistant/oasst1",
            "name": "OpenAssistant Conversations",
            "description": "Human-generated assistant conversations in multiple languages",
            "size": "~160,000 messages",
            "license": "Apache 2.0",
            "use_case": "Conversational AI, multi-turn dialogue"
        },
        {
            "id": "HuggingFaceH4/ultrachat_200k",
            "name": "UltraChat 200k",
            "description": "High-quality multi-turn dialogues for chat models",
            "size": "~200,000 conversations",
            "license": "MIT",
            "use_case": "Chat assistants, dialogue systems"
        },
    ],
    "Code & Programming": [
        {
            "id": "codeparrot/github-code",
            "name": "GitHub Code",
            "description": "Large collection of code from GitHub repositories",
            "size": "1TB+",
            "license": "Various",
            "use_case": "Code generation, completion"
        },
        {
            "id": "bigcode/starcoderdata",
            "name": "StarCoder Data",
            "description": "Cleaned code dataset used to train StarCoder",
            "size": "~783GB",
            "license": "Various open source",
            "use_case": "Code generation, understanding"
        },
        {
            "id": "iamtarun/python_code_instructions_18k_alpaca",
            "name": "Python Code Instructions",
            "description": "18k Python coding instructions in Alpaca format",
            "size": "~18,000 samples",
            "license": "Apache 2.0",
            "use_case": "Python code generation"
        },
    ],
    "Question Answering": [
        {
            "id": "squad_v2",
            "name": "SQuAD v2",
            "description": "Stanford Question Answering Dataset with unanswerable questions",
            "size": "~150,000 questions",
            "license": "CC BY-SA 4.0",
            "use_case": "Reading comprehension, Q&A"
        },
        {
            "id": "natural_questions",
            "name": "Natural Questions",
            "description": "Real Google search questions with Wikipedia answers",
            "size": "~300,000 questions",
            "license": "Apache 2.0",
            "use_case": "Open-domain Q&A"
        },
        {
            "id": "trivia_qa",
            "name": "TriviaQA",
            "description": "Large scale QA dataset with evidence documents",
            "size": "~95,000 questions",
            "license": "Apache 2.0",
            "use_case": "Trivia, factual Q&A"
        },
    ],
    "Summarization": [
        {
            "id": "cnn_dailymail",
            "name": "CNN/DailyMail",
            "description": "News articles with multi-sentence summaries",
            "size": "~300,000 articles",
            "license": "Apache 2.0",
            "use_case": "News summarization"
        },
        {
            "id": "xsum",
            "name": "XSum",
            "description": "BBC articles with single-sentence summaries",
            "size": "~226,000 articles",
            "license": "MIT",
            "use_case": "Extreme summarization"
        },
        {
            "id": "samsum",
            "name": "SAMSum",
            "description": "Messenger-like conversations with summaries",
            "size": "~16,000 conversations",
            "license": "CC BY-NC-ND 4.0",
            "use_case": "Dialogue summarization"
        },
    ],
    "Text Classification": [
        {
            "id": "imdb",
            "name": "IMDB Reviews",
            "description": "Movie reviews with sentiment labels",
            "size": "~50,000 reviews",
            "license": "Apache 2.0",
            "use_case": "Sentiment analysis"
        },
        {
            "id": "ag_news",
            "name": "AG News",
            "description": "News articles categorized by topic",
            "size": "~120,000 articles",
            "license": "Academic",
            "use_case": "Topic classification"
        },
        {
            "id": "emotion",
            "name": "Emotion Dataset",
            "description": "Twitter messages labeled with emotions",
            "size": "~20,000 tweets",
            "license": "Academic",
            "use_case": "Emotion detection"
        },
    ],
    "Reasoning & Math": [
        {
            "id": "gsm8k",
            "name": "GSM8K",
            "description": "Grade school math word problems with solutions",
            "size": "~8,500 problems",
            "license": "MIT",
            "use_case": "Math reasoning, chain-of-thought"
        },
        {
            "id": "lighteval/MATH",
            "name": "MATH Dataset",
            "description": "Challenging math problems from competitions",
            "size": "~12,500 problems",
            "license": "MIT",
            "use_case": "Advanced math reasoning"
        },
        {
            "id": "allenai/ai2_arc",
            "name": "AI2 ARC",
            "description": "Science exam questions requiring reasoning",
            "size": "~7,800 questions",
            "license": "CC BY-SA 4.0",
            "use_case": "Scientific reasoning"
        },
    ],
    "Books & Literature": [
        {
            "id": "bookcorpus",
            "name": "BookCorpus",
            "description": "11,000+ unpublished books across genres",
            "size": "~74M sentences",
            "license": "Academic",
            "use_case": "Language modeling, text understanding"
        },
        {
            "id": "pg19",
            "name": "PG-19 (Project Gutenberg)",
            "description": "Books from Project Gutenberg for long-form text",
            "size": "~28,000 books",
            "license": "Public Domain",
            "use_case": "Long-form text generation"
        },
    ],
    "Domain-Specific": [
        {
            "id": "pubmed_qa",
            "name": "PubMed QA",
            "description": "Biomedical research question answering",
            "size": "~211,000 questions",
            "license": "MIT",
            "use_case": "Medical/scientific Q&A"
        },
        {
            "id": "pile-of-law/pile-of-law",
            "name": "Pile of Law",
            "description": "Large legal text corpus",
            "size": "256GB",
            "license": "Public Domain",
            "use_case": "Legal text understanding"
        },
        {
            "id": "financial_phrasebank",
            "name": "Financial PhraseBank",
            "description": "Financial news with sentiment labels",
            "size": "~4,800 sentences",
            "license": "CC BY-NC-SA 3.0",
            "use_case": "Financial sentiment analysis"
        },
    ],
}


def render_hf_datasets_browser():
    """Render the HuggingFace datasets browser interface."""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d1b4e 100%); 
                padding: 1rem 1.25rem; border-radius: 8px; margin-bottom: 1rem;
                border-left: 4px solid #f59e0b;">
        <div style="color: #fcd34d; font-weight: 600; margin-bottom: 0.25rem;">🤗 HuggingFace Datasets</div>
        <div style="color: #e0e0e0; font-size: 0.9rem;">
            Browse and load curated datasets from HuggingFace Hub for fine-tuning your model.
            These datasets are pre-formatted and ready for training.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Two-column layout
    browse_col, preview_col = st.columns([1, 1], gap="medium")
    
    with browse_col:
        # Category selection
        st.markdown("""
        <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="color: #60a5fa; font-weight: 600; font-size: 0.9rem;">📂 Browse by Category</div>
        </div>
        """, unsafe_allow_html=True)
        
        category = st.selectbox(
            "Category",
            list(CURATED_DATASETS.keys()),
            label_visibility="collapsed"
        )
        
        # Show datasets in category
        datasets = CURATED_DATASETS[category]
        
        selected_dataset = None
        for ds in datasets:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{ds['name']}**")
                    st.caption(ds['description'][:80] + "..." if len(ds['description']) > 80 else ds['description'])
                with col2:
                    if st.button("Select", key=f"select_{ds['id']}", use_container_width=True):
                        st.session_state.selected_hf_dataset = ds
                        selected_dataset = ds
                st.markdown("<hr style='margin: 0.5rem 0; border-color: #3d4251;'>", unsafe_allow_html=True)
        
        # Custom dataset ID input
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="color: #60a5fa; font-weight: 600; font-size: 0.9rem;">🔍 Or Enter Dataset ID</div>
        </div>
        """, unsafe_allow_html=True)
        
        custom_id = st.text_input(
            "HuggingFace Dataset ID",
            placeholder="e.g., username/dataset-name",
            help="Enter any HuggingFace dataset ID"
        )
        
        if custom_id and st.button("🔍 Look Up Dataset", use_container_width=True):
            st.session_state.selected_hf_dataset = {
                "id": custom_id,
                "name": custom_id.split("/")[-1].replace("-", " ").title(),
                "description": "Custom dataset from HuggingFace Hub",
                "size": "Unknown",
                "license": "Check dataset page",
                "use_case": "Custom"
            }
    
    with preview_col:
        # Dataset preview
        st.markdown("""
        <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="color: #60a5fa; font-weight: 600; font-size: 0.9rem;">📋 Dataset Details</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get("selected_hf_dataset"):
            ds = st.session_state.selected_hf_dataset
            
            st.markdown(f"### {ds['name']}")
            st.markdown(f"**ID:** `{ds['id']}`")
            st.markdown(f"**Description:** {ds['description']}")
            st.markdown(f"**Size:** {ds['size']}")
            st.markdown(f"**License:** {ds['license']}")
            st.markdown(f"**Use Case:** {ds['use_case']}")
            
            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
            
            # Load options
            st.markdown("""
            <div style="background: #1a1d24; border: 1px solid #3d4251; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
                <div style="color: #f59e0b; font-weight: 600; font-size: 0.9rem;">⚙️ Load Options</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Split selection
            split = st.selectbox(
                "Dataset Split",
                ["train", "validation", "test", "all"],
                help="Which split of the dataset to load"
            )
            
            # Sample limit
            max_samples = st.number_input(
                "Max Samples",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                help="Limit samples to load (for large datasets)"
            )
            
            # Field mapping
            with st.expander("Field Mapping", expanded=False):
                st.caption("Map dataset fields to training format")
                instruction_field = st.text_input("Instruction Field", value="instruction", help="Field containing the instruction/prompt")
                input_field = st.text_input("Input Field", value="input", help="Field containing additional context (optional)")
                output_field = st.text_input("Output Field", value="output", help="Field containing the expected response")
            
            st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
            
            if st.button("📥 Load Dataset", type="primary", use_container_width=True):
                load_hf_dataset(
                    ds['id'],
                    split=split,
                    max_samples=max_samples,
                    field_mapping={
                        "instruction": instruction_field,
                        "input": input_field,
                        "output": output_field
                    }
                )
        else:
            st.markdown("""
            <div style="color: #6b7280; text-align: center; padding: 3rem; font-style: italic;">
                Select a dataset from the list or enter a custom dataset ID
            </div>
            """, unsafe_allow_html=True)


def load_hf_dataset(dataset_id: str, split: str = "train", 
                     max_samples: int = 1000, field_mapping: Dict[str, str] = None):
    """Load a dataset from HuggingFace Hub."""
    
    try:
        from datasets import load_dataset
        from core.dataset_handler import TrainingSample
        
        with st.spinner(f"Loading {dataset_id}..."):
            # Load dataset
            if split == "all":
                dataset = load_dataset(dataset_id)
                # Combine all splits
                all_samples = []
                for split_name in dataset.keys():
                    all_samples.extend(list(dataset[split_name]))
                data = all_samples[:max_samples]
            else:
                try:
                    dataset = load_dataset(dataset_id, split=split)
                    data = list(dataset)[:max_samples]
                except Exception:
                    # Try loading without split specification
                    dataset = load_dataset(dataset_id)
                    if split in dataset:
                        data = list(dataset[split])[:max_samples]
                    else:
                        # Use first available split
                        first_split = list(dataset.keys())[0]
                        data = list(dataset[first_split])[:max_samples]
                        st.warning(f"Split '{split}' not found. Using '{first_split}' instead.")
            
            # Convert to training samples
            samples = []
            field_mapping = field_mapping or {}
            instruction_field = field_mapping.get("instruction", "instruction")
            input_field = field_mapping.get("input", "input")
            output_field = field_mapping.get("output", "output")
            
            # Try to auto-detect fields if standard ones don't exist
            if data:
                sample_keys = list(data[0].keys())
                
                # Auto-detection fallbacks
                instruction_candidates = ["instruction", "prompt", "question", "text", "query", "input"]
                output_candidates = ["output", "response", "answer", "completion", "target", "label"]
                input_candidates = ["input", "context", "passage", "document", ""]
                
                if instruction_field not in sample_keys:
                    for candidate in instruction_candidates:
                        if candidate in sample_keys:
                            instruction_field = candidate
                            break
                
                if output_field not in sample_keys:
                    for candidate in output_candidates:
                        if candidate in sample_keys:
                            output_field = candidate
                            break
                
                if input_field not in sample_keys:
                    input_field = ""  # Optional field
                    for candidate in input_candidates:
                        if candidate in sample_keys and candidate != instruction_field:
                            input_field = candidate
                            break
            
            for item in data:
                try:
                    instruction = str(item.get(instruction_field, ""))
                    input_text = str(item.get(input_field, "")) if input_field else ""
                    output = str(item.get(output_field, ""))
                    
                    # Skip empty samples
                    if not instruction and not output:
                        continue
                    
                    # Handle cases where instruction is in a different format
                    if not instruction and "text" in item:
                        # Try to split text into instruction/output
                        text = str(item["text"])
                        if "###" in text:
                            parts = text.split("###")
                            instruction = parts[0].strip()
                            output = parts[-1].strip() if len(parts) > 1 else ""
                        else:
                            instruction = text
                            output = ""
                    
                    if instruction:  # Only add if we have an instruction
                        samples.append(TrainingSample(
                            instruction=instruction,
                            input=input_text,
                            output=output
                        ))
                except Exception as e:
                    continue  # Skip problematic samples
            
            if samples:
                st.session_state.training_samples = samples
                st.session_state.hf_dataset_loaded = dataset_id
                st.success(f"✅ Loaded {len(samples)} samples from {dataset_id}")
                st.rerun()
            else:
                st.error("Could not extract training samples from this dataset. Try adjusting field mapping.")
                
    except ImportError:
        st.error("datasets library required. Install with: pip install datasets")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.info("Try a different dataset or check the dataset ID.")


def search_hf_datasets(query: str) -> List[Dict[str, Any]]:
    """Search HuggingFace Hub for datasets."""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        datasets = api.list_datasets(search=query, limit=20)
        
        results = []
        for ds in datasets:
            results.append({
                "id": ds.id,
                "name": ds.id.split("/")[-1].replace("-", " ").title(),
                "description": ds.description or "No description available",
                "downloads": ds.downloads,
                "likes": ds.likes
            })
        
        return results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []
