"""
Dataset Handler for Processing Training Data.

Handles multiple input formats:
- JSON/JSONL: Alpaca format (instruction, input, output)
- Plain Text: Auto-chunk into training pairs
- PDF: Extract text with pypdf
- HTML/Confluence: Parse with BeautifulSoup
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Generator
from dataclasses import dataclass, field
import random

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics about a processed dataset."""
    total_samples: int
    train_samples: int
    validation_samples: int
    avg_input_length: float
    avg_output_length: float
    max_input_length: int
    max_output_length: int
    format_detected: str
    warnings: List[str] = field(default_factory=list)


@dataclass 
class TrainingSample:
    """A single training sample in standardized format."""
    instruction: str
    input: str
    output: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }
    
    def format_prompt(self, template: str = "alpaca") -> str:
        """Format the sample as a training prompt."""
        if template == "alpaca":
            if self.input:
                return (
                    f"### Instruction:\n{self.instruction}\n\n"
                    f"### Input:\n{self.input}\n\n"
                    f"### Response:\n{self.output}"
                )
            else:
                return (
                    f"### Instruction:\n{self.instruction}\n\n"
                    f"### Response:\n{self.output}"
                )
        elif template == "chatml":
            if self.input:
                return (
                    f"<|im_start|>user\n{self.instruction}\n\n{self.input}<|im_end|>\n"
                    f"<|im_start|>assistant\n{self.output}<|im_end|>"
                )
            else:
                return (
                    f"<|im_start|>user\n{self.instruction}<|im_end|>\n"
                    f"<|im_start|>assistant\n{self.output}<|im_end|>"
                )
        else:
            # Simple format
            return f"User: {self.instruction}\n{self.input}\n\nAssistant: {self.output}"


class DatasetHandler:
    """
    Handles loading, processing, and preparing datasets for fine-tuning.
    
    Supports multiple input formats and converts them all to a standardized
    format suitable for instruction-tuning.
    """
    
    SUPPORTED_FORMATS = {
        ".json": "json",
        ".jsonl": "jsonl", 
        ".csv": "csv",
        ".txt": "text",
        ".pdf": "pdf",
        ".html": "html",
        ".htm": "html",
    }
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the dataset handler with configuration."""
        self.config = self._load_config(config_path)
        self.uploads_path = Path(self.config["paths"]["data_uploads"])
        self.processed_path = Path(self.config["paths"]["data_processed"])
        
        # Ensure directories exist
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "paths": {
                "data_uploads": "./data/uploads",
                "data_processed": "./data/processed",
            },
            "dataset": {
                "validation_split": 0.1,
                "shuffle": True,
                "seed": 42,
            }
        }
    
    def detect_format(self, file_path: Path) -> str:
        """Detect the format of a data file."""
        suffix = file_path.suffix.lower()
        return self.SUPPORTED_FORMATS.get(suffix, "unknown")
    
    def load_file(self, file_path: Union[str, Path]) -> List[TrainingSample]:
        """
        Load a data file and convert to training samples.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of TrainingSample objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        format_type = self.detect_format(file_path)
        
        logger.info(f"Loading {format_type} file: {file_path}")
        
        if format_type == "json":
            return self._load_json(file_path)
        elif format_type == "jsonl":
            return self._load_jsonl(file_path)
        elif format_type == "csv":
            return self._load_csv(file_path)
        elif format_type == "text":
            return self._load_text(file_path)
        elif format_type == "pdf":
            return self._load_pdf(file_path)
        elif format_type == "html":
            return self._load_html(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_json(self, file_path: Path) -> List[TrainingSample]:
        """Load JSON file (expects Alpaca format or list of samples)."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both single object and list
        if isinstance(data, dict):
            data = [data]
        
        return self._parse_samples(data)
    
    def _load_jsonl(self, file_path: Path) -> List[TrainingSample]:
        """Load JSONL file (one JSON object per line)."""
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        samples.append(obj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        return self._parse_samples(samples)
    
    def _load_csv(self, file_path: Path) -> List[TrainingSample]:
        """Load CSV file with instruction/input/output columns."""
        df = pd.read_csv(file_path)
        
        # Try to map columns to our format
        column_mapping = self._detect_csv_columns(df.columns.tolist())
        
        samples = []
        for _, row in df.iterrows():
            sample = TrainingSample(
                instruction=str(row.get(column_mapping.get("instruction", "instruction"), "")),
                input=str(row.get(column_mapping.get("input", "input"), "")),
                output=str(row.get(column_mapping.get("output", "output"), ""))
            )
            if sample.instruction or sample.output:
                samples.append(sample)
        
        return samples
    
    def _detect_csv_columns(self, columns: List[str]) -> Dict[str, str]:
        """Detect which columns map to instruction/input/output."""
        mapping = {}
        columns_lower = [c.lower() for c in columns]
        
        # Instruction column detection
        for pattern in ["instruction", "prompt", "question", "query"]:
            for i, col in enumerate(columns_lower):
                if pattern in col:
                    mapping["instruction"] = columns[i]
                    break
            if "instruction" in mapping:
                break
        
        # Input column detection
        for pattern in ["input", "context", "text"]:
            for i, col in enumerate(columns_lower):
                if pattern in col and columns[i] != mapping.get("instruction"):
                    mapping["input"] = columns[i]
                    break
            if "input" in mapping:
                break
        
        # Output column detection
        for pattern in ["output", "response", "answer", "completion"]:
            for i, col in enumerate(columns_lower):
                if pattern in col:
                    mapping["output"] = columns[i]
                    break
            if "output" in mapping:
                break
        
        return mapping
    
    def _load_text(self, file_path: Path) -> List[TrainingSample]:
        """
        Load plain text file and convert to training samples.
        
        Attempts to detect structure (Q&A pairs, sections, etc.)
        Falls back to chunking if no structure detected.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        samples = []
        
        # Try to detect Q&A format
        qa_patterns = [
            r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:|$)",
            r"Question:\s*(.+?)\nAnswer:\s*(.+?)(?=\nQuestion:|$)",
            r"\?\s*\n(.+?)(?=\n\n|\n\?|$)",
        ]
        
        for pattern in qa_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    if len(match) >= 2:
                        samples.append(TrainingSample(
                            instruction=match[0].strip(),
                            input="",
                            output=match[1].strip()
                        ))
                break
        
        # If no Q&A found, chunk the text
        if not samples:
            samples = self._chunk_text(content)
        
        return samples
    
    def _chunk_text(self, content: str, chunk_size: int = 500) -> List[TrainingSample]:
        """
        Chunk text into training samples for knowledge injection.
        
        Creates instruction-following pairs from text chunks.
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        samples = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para.split())
            
            if current_size + para_size > chunk_size and current_chunk:
                # Create a sample from current chunk
                chunk_text = "\n\n".join(current_chunk)
                samples.append(TrainingSample(
                    instruction="Summarize and explain the following information:",
                    input=chunk_text,
                    output=f"Based on the provided information: {chunk_text[:200]}..."
                ))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        
        # Handle remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            samples.append(TrainingSample(
                instruction="Summarize and explain the following information:",
                input=chunk_text,
                output=f"Based on the provided information: {chunk_text[:200]}..."
            ))
        
        return samples
    
    def _load_pdf(self, file_path: Path) -> List[TrainingSample]:
        """Load PDF file and extract text for training."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF processing. "
                "Install with: pip install pypdf"
            )
        
        reader = PdfReader(file_path)
        
        all_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
        
        content = "\n\n".join(all_text)
        
        # Use text processing
        return self._chunk_text(content)
    
    def _load_html(self, file_path: Path) -> List[TrainingSample]:
        """Load HTML file (including Confluence exports) and extract content."""
        try:
            from bs4 import BeautifulSoup
            import html2text
        except ImportError:
            raise ImportError(
                "beautifulsoup4 and html2text are required for HTML processing. "
                "Install with: pip install beautifulsoup4 html2text"
            )
        
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Convert to markdown-like text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        
        text = h.handle(str(soup))
        
        # Use text processing
        return self._chunk_text(text)
    
    def _parse_samples(self, data: List[Dict]) -> List[TrainingSample]:
        """Parse a list of dictionaries into TrainingSample objects."""
        samples = []
        
        for item in data:
            # Handle various field names
            instruction = (
                item.get("instruction") or 
                item.get("prompt") or 
                item.get("question") or 
                ""
            )
            
            input_text = (
                item.get("input") or 
                item.get("context") or 
                ""
            )
            
            output = (
                item.get("output") or 
                item.get("response") or 
                item.get("answer") or 
                item.get("completion") or 
                ""
            )
            
            # Handle conversation format
            if "conversations" in item:
                conversations = item["conversations"]
                for i in range(0, len(conversations) - 1, 2):
                    if i + 1 < len(conversations):
                        samples.append(TrainingSample(
                            instruction=conversations[i].get("value", ""),
                            input="",
                            output=conversations[i + 1].get("value", "")
                        ))
                continue
            
            if instruction or output:
                samples.append(TrainingSample(
                    instruction=instruction,
                    input=input_text,
                    output=output
                ))
        
        return samples
    
    def prepare_dataset(self, 
                        samples: List[TrainingSample],
                        validation_split: Optional[float] = None,
                        shuffle: bool = True,
                        seed: int = 42) -> Dict[str, List[TrainingSample]]:
        """
        Prepare dataset for training with train/validation split.
        
        Args:
            samples: List of training samples
            validation_split: Fraction for validation (default from config)
            shuffle: Whether to shuffle data
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train' and 'validation' sample lists
        """
        if validation_split is None:
            validation_split = self.config["dataset"]["validation_split"]
        
        if shuffle:
            random.seed(seed)
            samples = samples.copy()
            random.shuffle(samples)
        
        split_idx = int(len(samples) * (1 - validation_split))
        
        return {
            "train": samples[:split_idx],
            "validation": samples[split_idx:]
        }
    
    def get_statistics(self, samples: List[TrainingSample]) -> DatasetStats:
        """Calculate statistics about the dataset."""
        if not samples:
            return DatasetStats(
                total_samples=0,
                train_samples=0,
                validation_samples=0,
                avg_input_length=0,
                avg_output_length=0,
                max_input_length=0,
                max_output_length=0,
                format_detected="empty",
                warnings=["No samples loaded"]
            )
        
        input_lengths = []
        output_lengths = []
        warnings = []
        
        for sample in samples:
            combined_input = f"{sample.instruction} {sample.input}".strip()
            input_lengths.append(len(combined_input.split()))
            output_lengths.append(len(sample.output.split()))
        
        # Check for potential issues
        if max(input_lengths) > 500:
            warnings.append(f"Some inputs are long ({max(input_lengths)} words). Consider truncation.")
        
        if min(output_lengths) < 5:
            warnings.append("Some outputs are very short. Check data quality.")
        
        prepared = self.prepare_dataset(samples)
        
        return DatasetStats(
            total_samples=len(samples),
            train_samples=len(prepared["train"]),
            validation_samples=len(prepared["validation"]),
            avg_input_length=sum(input_lengths) / len(input_lengths),
            avg_output_length=sum(output_lengths) / len(output_lengths),
            max_input_length=max(input_lengths),
            max_output_length=max(output_lengths),
            format_detected="alpaca",
            warnings=warnings
        )
    
    def to_hf_dataset(self, samples: List[TrainingSample], 
                       prompt_template: str = "alpaca"):
        """
        Convert samples to HuggingFace Dataset format.
        
        Args:
            samples: List of training samples
            prompt_template: Template for formatting prompts
            
        Returns:
            HuggingFace Dataset object
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "datasets library is required. "
                "Install with: pip install datasets"
            )
        
        # Format as full prompts
        formatted_data = []
        for sample in samples:
            formatted_data.append({
                "text": sample.format_prompt(prompt_template),
                "instruction": sample.instruction,
                "input": sample.input,
                "output": sample.output,
            })
        
        return Dataset.from_list(formatted_data)
    
    def save_processed(self, samples: List[TrainingSample], 
                       output_name: str) -> Path:
        """Save processed samples to JSONL file."""
        output_path = self.processed_path / f"{output_name}.jsonl"
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                json.dump(sample.to_dict(), f)
                f.write("\n")
        
        logger.info(f"Saved {len(samples)} samples to {output_path}")
        return output_path


# Example IT Support Templates
IT_SUPPORT_TEMPLATES = {
    "servicenow_ticket": {
        "instruction": "Analyze this ServiceNow ticket and provide a resolution summary",
        "input_template": "Ticket: {ticket_id}\nCategory: {category}\nDescription: {description}",
        "output_template": "Summary: {summary}\n\nResolution Steps:\n{steps}"
    },
    "knowledge_article": {
        "instruction": "Based on this knowledge article, answer the user's question",
        "input_template": "Article: {title}\n\n{content}\n\nQuestion: {question}",
        "output_template": "{answer}"
    },
    "sop": {
        "instruction": "Explain how to perform this procedure based on the SOP",
        "input_template": "SOP: {title}\n\nProcedure: {question}",
        "output_template": "{steps}"
    }
}


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    handler = DatasetHandler()
    
    # Test with a simple example
    test_data = [
        {
            "instruction": "Summarize this ServiceNow ticket",
            "input": "INC0012345: User cannot access VPN. Error: Connection timed out.",
            "output": "VPN connectivity issue requiring network troubleshooting."
        },
        {
            "instruction": "How do I reset my password?",
            "input": "",
            "output": "Navigate to password.company.com and click 'Forgot Password'."
        }
    ]
    
    # Save test data
    test_file = Path("./data/uploads/test_data.json")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=2)
    
    # Load and process
    samples = handler.load_file(test_file)
    print(f"\nLoaded {len(samples)} samples")
    
    stats = handler.get_statistics(samples)
    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples: {stats.total_samples}")
    print(f"Train/Val split: {stats.train_samples}/{stats.validation_samples}")
    print(f"Avg input length: {stats.avg_input_length:.1f} words")
    print(f"Avg output length: {stats.avg_output_length:.1f} words")
    
    if stats.warnings:
        print("\nWarnings:")
        for warning in stats.warnings:
            print(f"  - {warning}")
    
    # Show formatted sample
    print("\n=== Sample Formatted Prompt ===")
    print(samples[0].format_prompt("alpaca"))
