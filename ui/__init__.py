"""
UI module for LLM Fine-Tuning Platform.

This package contains:
- pages/: Individual page components for each step
- components/: Reusable UI components
"""

from pathlib import Path

# UI version
__version__ = "0.1.0"

# Paths
UI_DIR = Path(__file__).parent
PAGES_DIR = UI_DIR / "pages"
COMPONENTS_DIR = UI_DIR / "components"
