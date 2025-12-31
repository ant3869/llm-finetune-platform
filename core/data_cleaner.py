"""
Advanced Data Cleaning Module for LLM Fine-Tuning.

Provides comprehensive text cleaning, normalization, and preprocessing
for training data quality improvement.

Features:
- Prefix/suffix/delimiter removal
- Pattern-based text extraction
- Unicode normalization
- HTML/markdown stripping
- Deduplication
- Quality filtering
- Automated cleaning pipelines
"""

import re
import unicodedata
import logging
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    # Delimiter removal
    remove_line_prefix: Optional[str] = None  # e.g., "XX |" to remove from each line
    remove_line_suffix: Optional[str] = None
    prefix_delimiter: Optional[str] = None  # Split on this and take right part
    suffix_delimiter: Optional[str] = None  # Split on this and take left part
    
    # Pattern removal
    patterns_to_remove: List[str] = field(default_factory=list)  # Regex patterns
    patterns_to_replace: Dict[str, str] = field(default_factory=dict)  # pattern: replacement
    
    # Text normalization
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    normalize_quotes: bool = True
    lowercase: bool = False
    
    # Content filtering
    remove_html_tags: bool = True
    remove_markdown: bool = False
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    
    # Quality filters
    min_length: int = 10
    max_length: int = 100000
    min_word_count: int = 3
    max_repetition_ratio: float = 0.5  # Max ratio of repeated words
    remove_duplicates: bool = True
    
    # Advanced
    remove_boilerplate: bool = False  # Remove common boilerplate text
    fix_encoding: bool = True
    strip_lines: bool = True


class TextCleaner:
    """
    Text cleaning utilities for preprocessing training data.
    """
    
    # Common boilerplate patterns
    BOILERPLATE_PATTERNS = [
        r"^(page \d+|chapter \d+)[\s:]*",
        r"^(confidential|internal use only|draft)[\s:]*",
        r"copyright \d{4}.*$",
        r"all rights reserved.*$",
        r"^(created|modified|updated):\s*\d{4}.*$",
        r"^\[?\d+\]?\s*$",  # Just page numbers
    ]
    
    # HTML tag pattern
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    
    # URL pattern
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|'
        r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    )
    
    # Email pattern
    EMAIL_PATTERN = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
    
    # Phone pattern (various formats)
    PHONE_PATTERN = re.compile(
        r'(\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    )
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """Initialize cleaner with configuration."""
        self.config = config or CleaningConfig()
        self._seen_hashes: Set[str] = set()
    
    def clean_text(self, text: str) -> str:
        """
        Apply all configured cleaning operations to text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Fix encoding issues first
        if self.config.fix_encoding:
            text = self._fix_encoding(text)
        
        # Remove line prefixes/suffixes
        if self.config.remove_line_prefix or self.config.remove_line_suffix:
            text = self._remove_line_affixes(text)
        
        # Handle delimiters
        if self.config.prefix_delimiter:
            text = self._remove_prefix_by_delimiter(text, self.config.prefix_delimiter)
        if self.config.suffix_delimiter:
            text = self._remove_suffix_by_delimiter(text, self.config.suffix_delimiter)
        
        # Remove patterns
        for pattern in self.config.patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Replace patterns
        for pattern, replacement in self.config.patterns_to_replace.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove HTML tags
        if self.config.remove_html_tags:
            text = self._remove_html(text)
        
        # Remove markdown formatting
        if self.config.remove_markdown:
            text = self._remove_markdown(text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = self.URL_PATTERN.sub("", text)
        
        # Remove emails
        if self.config.remove_emails:
            text = self.EMAIL_PATTERN.sub("", text)
        
        # Remove phone numbers
        if self.config.remove_phone_numbers:
            text = self.PHONE_PATTERN.sub("", text)
        
        # Remove boilerplate
        if self.config.remove_boilerplate:
            text = self._remove_boilerplate(text)
        
        # Normalize unicode
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # Normalize quotes
        if self.config.normalize_quotes:
            text = self._normalize_quotes(text)
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Strip lines
        if self.config.strip_lines:
            lines = [line.strip() for line in text.split('\n')]
            text = '\n'.join(lines)
        
        return text.strip()
    
    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Common mojibake fixes
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'â€¦': '...',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            'Ã¢': 'â',
            'Ã®': 'î',
            'Ã´': 'ô',
            'Ã»': 'û',
            'Ã§': 'ç',
            '\x00': '',  # Null bytes
            '\ufeff': '',  # BOM
        }
        
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        return text
    
    def _remove_line_affixes(self, text: str) -> str:
        """Remove prefix/suffix from each line."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if self.config.remove_line_prefix and line.startswith(self.config.remove_line_prefix):
                line = line[len(self.config.remove_line_prefix):]
            if self.config.remove_line_suffix and line.endswith(self.config.remove_line_suffix):
                line = line[:-len(self.config.remove_line_suffix)]
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_prefix_by_delimiter(self, text: str, delimiter: str) -> str:
        """Remove everything before delimiter on each line."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if delimiter in line:
                parts = line.split(delimiter, 1)
                if len(parts) > 1:
                    line = parts[1]
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_suffix_by_delimiter(self, text: str, delimiter: str) -> str:
        """Remove everything after delimiter on each line."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if delimiter in line:
                parts = line.rsplit(delimiter, 1)
                line = parts[0]
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and decode entities."""
        import html
        text = self.HTML_TAG_PATTERN.sub(' ', text)
        text = html.unescape(text)
        return text
    
    def _remove_markdown(self, text: str) -> str:
        """Remove markdown formatting."""
        # Headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Bold/italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Links
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Lists
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        return text
    
    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate text."""
        for pattern in self.BOILERPLATE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # NFKC normalization - compatibility decomposition followed by canonical composition
        text = unicodedata.normalize('NFKC', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace various whitespace with standard space
        text = re.sub(r'[\t\r\f\v]+', ' ', text)
        
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Collapse multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize various quote characters to standard ones."""
        # Normalize single quotes
        text = re.sub(r'[''‚‛]', "'", text)
        
        # Normalize double quotes
        text = re.sub(r'[""„‟]', '"', text)
        
        return text
    
    def passes_quality_filter(self, text: str) -> Tuple[bool, str]:
        """
        Check if text passes quality filters.
        
        Returns:
            Tuple of (passes, reason if failed)
        """
        # Length check
        if len(text) < self.config.min_length:
            return False, f"Too short ({len(text)} < {self.config.min_length})"
        
        if len(text) > self.config.max_length:
            return False, f"Too long ({len(text)} > {self.config.max_length})"
        
        # Word count check
        words = text.split()
        if len(words) < self.config.min_word_count:
            return False, f"Too few words ({len(words)} < {self.config.min_word_count})"
        
        # Repetition check
        if words:
            word_counts = Counter(words)
            most_common_count = word_counts.most_common(1)[0][1]
            repetition_ratio = most_common_count / len(words)
            
            if repetition_ratio > self.config.max_repetition_ratio:
                return False, f"Too repetitive (ratio: {repetition_ratio:.2f})"
        
        # Duplicate check
        if self.config.remove_duplicates:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._seen_hashes:
                return False, "Duplicate content"
            self._seen_hashes.add(text_hash)
        
        return True, ""
    
    def reset_dedup(self):
        """Reset deduplication state."""
        self._seen_hashes.clear()


class DataCleaningPipeline:
    """
    Pipeline for batch cleaning of training samples.
    """
    
    # Preset cleaning configurations
    PRESETS = {
        "minimal": CleaningConfig(
            normalize_unicode=True,
            normalize_whitespace=True,
            strip_lines=True,
        ),
        "standard": CleaningConfig(
            normalize_unicode=True,
            normalize_whitespace=True,
            normalize_quotes=True,
            remove_html_tags=True,
            strip_lines=True,
            min_length=10,
        ),
        "aggressive": CleaningConfig(
            normalize_unicode=True,
            normalize_whitespace=True,
            normalize_quotes=True,
            remove_html_tags=True,
            remove_markdown=True,
            remove_urls=True,
            remove_emails=True,
            remove_boilerplate=True,
            strip_lines=True,
            min_length=20,
            remove_duplicates=True,
        ),
        "it_support": CleaningConfig(
            normalize_unicode=True,
            normalize_whitespace=True,
            normalize_quotes=True,
            remove_html_tags=True,
            strip_lines=True,
            min_length=10,
            patterns_to_remove=[
                r"^(ticket|case|incident)\s*#?\d+[\s:]*",  # Ticket numbers at start
                r"\b(created|updated|resolved)\s*:\s*\d{1,2}/\d{1,2}/\d{2,4}.*$",  # Date stamps
            ],
        ),
    }
    
    def __init__(self, config: Optional[CleaningConfig] = None, preset: str = None):
        """
        Initialize cleaning pipeline.
        
        Args:
            config: Custom cleaning configuration
            preset: Use a preset configuration ("minimal", "standard", "aggressive", "it_support")
        """
        if preset and preset in self.PRESETS:
            self.config = self.PRESETS[preset]
        else:
            self.config = config or CleaningConfig()
        
        self.cleaner = TextCleaner(self.config)
        self.stats = {
            "processed": 0,
            "passed": 0,
            "filtered": 0,
            "filter_reasons": Counter(),
        }
    
    def clean_samples(self, samples: List[Any], 
                      instruction_field: str = "instruction",
                      input_field: str = "input", 
                      output_field: str = "output") -> List[Any]:
        """
        Clean a list of training samples.
        
        Args:
            samples: List of sample objects (TrainingSample or dicts)
            instruction_field: Field name for instruction
            input_field: Field name for input
            output_field: Field name for output
            
        Returns:
            List of cleaned samples that passed quality filters
        """
        from core.dataset_handler import TrainingSample
        
        cleaned_samples = []
        self.stats = {"processed": 0, "passed": 0, "filtered": 0, "filter_reasons": Counter()}
        self.cleaner.reset_dedup()
        
        for sample in samples:
            self.stats["processed"] += 1
            
            # Extract fields
            if hasattr(sample, instruction_field):
                instruction = getattr(sample, instruction_field)
                input_text = getattr(sample, input_field, "")
                output = getattr(sample, output_field)
            else:
                instruction = sample.get(instruction_field, "")
                input_text = sample.get(input_field, "")
                output = sample.get(output_field, "")
            
            # Clean each field
            cleaned_instruction = self.cleaner.clean_text(instruction)
            cleaned_input = self.cleaner.clean_text(input_text) if input_text else ""
            cleaned_output = self.cleaner.clean_text(output)
            
            # Validate output (most important for training)
            passes, reason = self.cleaner.passes_quality_filter(cleaned_output)
            
            if not passes:
                self.stats["filtered"] += 1
                self.stats["filter_reasons"][reason] += 1
                continue
            
            # Also check instruction isn't empty
            if not cleaned_instruction.strip():
                self.stats["filtered"] += 1
                self.stats["filter_reasons"]["Empty instruction"] += 1
                continue
            
            # Create cleaned sample
            cleaned_sample = TrainingSample(
                instruction=cleaned_instruction,
                input=cleaned_input,
                output=cleaned_output
            )
            cleaned_samples.append(cleaned_sample)
            self.stats["passed"] += 1
        
        return cleaned_samples
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics."""
        return {
            **self.stats,
            "pass_rate": self.stats["passed"] / max(self.stats["processed"], 1),
        }


def create_delimiter_cleaner(delimiter: str, keep: str = "right") -> CleaningConfig:
    """
    Create a cleaning config for delimiter-based text removal.
    
    Args:
        delimiter: The delimiter string (e.g., "XX |", " | ", ":")
        keep: Which part to keep ("left" or "right")
        
    Returns:
        CleaningConfig for the specified delimiter removal
    
    Example:
        # Remove "XX |" prefix from lines like "XX | Actual content here"
        config = create_delimiter_cleaner("XX |", keep="right")
    """
    if keep == "right":
        return CleaningConfig(
            prefix_delimiter=delimiter,
            normalize_whitespace=True,
            strip_lines=True,
        )
    else:
        return CleaningConfig(
            suffix_delimiter=delimiter,
            normalize_whitespace=True,
            strip_lines=True,
        )


def create_pattern_cleaner(patterns: List[str], replacements: Dict[str, str] = None) -> CleaningConfig:
    """
    Create a cleaning config for pattern-based text removal/replacement.
    
    Args:
        patterns: List of regex patterns to remove
        replacements: Dict of pattern -> replacement mappings
        
    Returns:
        CleaningConfig for pattern-based cleaning
    
    Example:
        # Remove timestamps and ticket numbers
        config = create_pattern_cleaner(
            patterns=[r"\d{4}-\d{2}-\d{2}", r"INC\d+"],
            replacements={r"\s+": " "}  # Collapse whitespace
        )
    """
    return CleaningConfig(
        patterns_to_remove=patterns,
        patterns_to_replace=replacements or {},
        normalize_whitespace=True,
        strip_lines=True,
    )


# Convenience functions for common cleaning tasks
def clean_it_ticket_data(text: str) -> str:
    """Clean IT ticket/ServiceNow data."""
    config = CleaningConfig(
        normalize_unicode=True,
        normalize_whitespace=True,
        remove_html_tags=True,
        patterns_to_remove=[
            r"^(ticket|case|incident)\s*#?\s*\d+[\s:]*",
            r"\[(created|updated|assigned)\s*:\s*[^\]]+\]",
            r"sys_id:\s*\w+",
            r"^-{3,}$",  # Separator lines
        ],
        strip_lines=True,
    )
    cleaner = TextCleaner(config)
    return cleaner.clean_text(text)


def remove_line_prefixes(text: str, prefix: str) -> str:
    """Simple utility to remove a prefix from each line."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if prefix in line:
            idx = line.find(prefix)
            line = line[idx + len(prefix):]
        cleaned.append(line.strip())
    return '\n'.join(cleaned)


def remove_by_delimiter(text: str, delimiter: str, keep: str = "right") -> str:
    """
    Remove text before/after delimiter on each line.
    
    Args:
        text: Input text
        delimiter: Delimiter string
        keep: "left" or "right" - which part to keep
        
    Returns:
        Cleaned text
    """
    lines = text.split('\n')
    cleaned = []
    
    for line in lines:
        if delimiter in line:
            parts = line.split(delimiter, 1)
            if keep == "right" and len(parts) > 1:
                line = parts[1]
            elif keep == "left":
                line = parts[0]
        cleaned.append(line.strip())
    
    return '\n'.join(cleaned)
