"""
UI module for LLM Fine-Tuning Platform.

This package contains:
- pages/: Individual page components for each step
- components/: Reusable UI components
- styles.py: Custom CSS and styling
- help_system.py: Contextual help tooltips
"""

from pathlib import Path

# UI version
__version__ = "0.2.0"

# Paths
UI_DIR = Path(__file__).parent
PAGES_DIR = UI_DIR / "pages"
COMPONENTS_DIR = UI_DIR / "components"

# Import main styling functions
from .styles import apply_theme, inject_custom_css, spacer
from .help_system import init_help_system, render_help_footer, with_help, help_label
