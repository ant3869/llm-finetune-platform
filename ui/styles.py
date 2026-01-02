"""
Custom styling module for LLM Fine-Tuning Platform.
Provides Tailwind-inspired CSS and component styling for Streamlit.
"""

import streamlit as st

# Color palette (Tailwind-inspired)
COLORS = {
    # Primary colors
    "primary-50": "#eff6ff",
    "primary-100": "#dbeafe", 
    "primary-200": "#bfdbfe",
    "primary-300": "#93c5fd",
    "primary-400": "#60a5fa",
    "primary-500": "#3b82f6",
    "primary-600": "#2563eb",
    "primary-700": "#1d4ed8",
    "primary-800": "#1e40af",
    "primary-900": "#1e3a8a",
    
    # Success/Green
    "success-50": "#f0fdf4",
    "success-500": "#22c55e",
    "success-600": "#16a34a",
    "success-700": "#15803d",
    
    # Warning/Amber
    "warning-50": "#fffbeb",
    "warning-500": "#f59e0b",
    "warning-600": "#d97706",
    
    # Error/Red
    "error-50": "#fef2f2",
    "error-500": "#ef4444",
    "error-600": "#dc2626",
    
    # Neutral/Gray
    "gray-50": "#f9fafb",
    "gray-100": "#f3f4f6",
    "gray-200": "#e5e7eb",
    "gray-300": "#d1d5db",
    "gray-400": "#9ca3af",
    "gray-500": "#6b7280",
    "gray-600": "#4b5563",
    "gray-700": "#374151",
    "gray-800": "#1f2937",
    "gray-900": "#111827",
}


def inject_custom_css():
    """Inject custom CSS styling into the Streamlit app."""
    st.markdown(get_global_styles(), unsafe_allow_html=True)


def get_global_styles() -> str:
    """Return global CSS styles for the app."""
    return f"""
    <style>
        /* ============================================
           GLOBAL STYLES - Tailwind-inspired Design
           Supports both Light and Dark modes
           ============================================ */
        
        /* Light mode variables (default) */
        :root {{
            --primary: {COLORS['primary-500']};
            --primary-hover: {COLORS['primary-600']};
            --primary-light: {COLORS['primary-100']};
            --success: {COLORS['success-500']};
            --warning: {COLORS['warning-500']};
            --error: {COLORS['error-500']};
            --bg-primary: #ffffff;
            --bg-secondary: {COLORS['gray-50']};
            --bg-tertiary: {COLORS['gray-100']};
            --text-primary: {COLORS['gray-900']};
            --text-secondary: {COLORS['gray-700']};
            --text-muted: {COLORS['gray-500']};
            --border-color: {COLORS['gray-200']};
            --border-hover: {COLORS['gray-300']};
            --card-bg: #ffffff;
            --card-shadow: rgba(0, 0, 0, 0.1);
            --border-radius: 0.5rem;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }}

        /* Dark mode variables - detect Streamlit's dark theme */
        [data-testid="stAppViewContainer"][data-theme="dark"],
        .stApp[data-theme="dark"],
        [data-theme="dark"] {{
            --primary: {COLORS['primary-400']};
            --primary-hover: {COLORS['primary-300']};
            --primary-light: rgba(59, 130, 246, 0.2);
            --bg-primary: #0e1117;
            --bg-secondary: #1a1d24;
            --bg-tertiary: #262730;
            --text-primary: #fafafa;
            --text-secondary: #e0e0e0;
            --text-muted: #9ca3af;
            --border-color: #3d4251;
            --border-hover: #4d5263;
            --card-bg: #1a1d24;
            --card-shadow: rgba(0, 0, 0, 0.3);
        }}

        /* Also handle via media query for system preference */
        @media (prefers-color-scheme: dark) {{
            :root {{
                --primary: {COLORS['primary-400']};
                --primary-hover: {COLORS['primary-300']};
                --primary-light: rgba(59, 130, 246, 0.2);
                --bg-primary: #0e1117;
                --bg-secondary: #1a1d24;
                --bg-tertiary: #262730;
                --text-primary: #fafafa;
                --text-secondary: #e0e0e0;
                --text-muted: #9ca3af;
                --border-color: #3d4251;
                --border-hover: #4d5263;
                --card-bg: #1a1d24;
                --card-shadow: rgba(0, 0, 0, 0.3);
            }}
        }}

        /* Smooth transitions everywhere */
        * {{
            transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out, color 0.2s ease-in-out;
        }}

        /* Main container */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}

        /* Headers - use CSS variables for dark mode support */
        h1 {{
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            color: var(--text-primary) !important;
        }}

        h2 {{
            font-weight: 600 !important;
            color: var(--text-secondary) !important;
        }}

        h3 {{
            font-weight: 600 !important;
            color: var(--text-secondary) !important;
        }}

        /* Cards/Containers */
        div[data-testid="stExpander"] {{
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            background: var(--card-bg) !important;
        }}

        div[data-testid="stExpander"]:hover {{
            box-shadow: var(--shadow-md);
            border-color: var(--primary) !important;
        }}

        /* Primary buttons */
        .stButton>button[kind="primary"],
        .stButton>button[data-testid="baseButton-primary"] {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%) !important;
            border: none !important;
            border-radius: var(--border-radius) !important;
            padding: 0.625rem 1.25rem !important;
            font-weight: 600 !important;
            box-shadow: var(--shadow-md);
            color: white !important;
        }}

        .stButton>button[kind="primary"]:hover,
        .stButton>button[data-testid="baseButton-primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: var(--shadow-lg);
        }}

        /* Secondary/default buttons */
        .stButton>button[kind="secondary"],
        .stButton>button[data-testid="baseButton-secondary"] {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
        }}

        .stButton>button[kind="secondary"]:hover,
        .stButton>button[data-testid="baseButton-secondary"]:hover {{
            background: var(--bg-tertiary) !important;
            border-color: var(--primary) !important;
            color: var(--primary) !important;
        }}

        /* Input fields */
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>div {{
            border-radius: var(--border-radius) !important;
            border: 1px solid var(--border-color) !important;
            background: var(--card-bg) !important;
            color: var(--text-primary) !important;
        }}

        .stTextInput>div>div>input:focus,
        .stTextArea>div>div>textarea:focus {{
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px var(--primary-light) !important;
        }}

        /* Tabs - FIXED FOR DARK MODE */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            background: transparent;
            border-bottom: 1px solid var(--border-color);
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: var(--border-radius) var(--border-radius) 0 0;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            background: var(--bg-tertiary) !important;
            border: 1px solid var(--border-color) !important;
            border-bottom: none !important;
            color: var(--text-secondary) !important;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            background: var(--bg-secondary) !important;
            color: var(--primary) !important;
        }}

        .stTabs [aria-selected="true"] {{
            background: var(--card-bg) !important;
            border-color: var(--primary) !important;
            border-bottom: 2px solid var(--card-bg) !important;
            color: var(--primary) !important;
        }}

        /* Tab panel background */
        .stTabs [data-baseweb="tab-panel"] {{
            background: transparent;
        }}

        /* Metrics */
        div[data-testid="stMetric"] {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: 1rem;
            box-shadow: var(--shadow-sm);
        }}

        div[data-testid="stMetric"]:hover {{
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }}

        div[data-testid="stMetricValue"] {{
            font-weight: 700 !important;
            color: var(--primary) !important;
        }}

        div[data-testid="stMetricLabel"] {{
            color: var(--text-secondary) !important;
        }}

        /* Progress bar */
        .stProgress>div>div>div>div {{
            background: linear-gradient(90deg, var(--primary) 0%, {COLORS['primary-400']} 100%) !important;
            border-radius: 9999px;
        }}

        /* Success/Info/Warning/Error messages - dark mode compatible */
        .stSuccess, div[data-testid="stAlert"][data-baseweb-type="positive"] {{
            background: rgba(34, 197, 94, 0.15) !important;
            border-left: 4px solid var(--success) !important;
            border-radius: var(--border-radius) !important;
            color: var(--text-primary) !important;
        }}

        .stInfo, div[data-testid="stAlert"][data-baseweb-type="info"] {{
            background: var(--primary-light) !important;
            border-left: 4px solid var(--primary) !important;
            border-radius: var(--border-radius) !important;
            color: var(--text-primary) !important;
        }}

        .stWarning, div[data-testid="stAlert"][data-baseweb-type="warning"] {{
            background: rgba(245, 158, 11, 0.15) !important;
            border-left: 4px solid var(--warning) !important;
            border-radius: var(--border-radius) !important;
            color: var(--text-primary) !important;
        }}

        .stError, div[data-testid="stAlert"][data-baseweb-type="negative"] {{
            background: rgba(239, 68, 68, 0.15) !important;
            border-left: 4px solid var(--error) !important;
            border-radius: var(--border-radius) !important;
            color: var(--text-primary) !important;
        }}

        /* Dataframe styling */
        div[data-testid="stDataFrame"] {{
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-color);
        }}

        section[data-testid="stSidebar"] .stButton>button {{
            width: 100%;
            justify-content: flex-start;
            padding-left: 1rem;
        }}

        /* Sidebar text */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: var(--text-primary) !important;
        }}

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {{
            color: var(--text-secondary) !important;
        }}

        /* Radio buttons */
        .stRadio>div {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .stRadio>div>label {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: 0.5rem 1rem;
            cursor: pointer;
            color: var(--text-secondary);
        }}

        .stRadio>div>label:hover {{
            border-color: var(--primary);
            background: var(--primary-light);
        }}

        /* Divider */
        hr {{
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 1.5rem 0;
        }}

        /* Code blocks */
        .stCodeBlock {{
            border-radius: var(--border-radius) !important;
            box-shadow: var(--shadow-sm);
        }}

        /* Slider */
        .stSlider>div>div>div>div {{
            background: var(--primary) !important;
        }}

        /* Checkbox */
        .stCheckbox>label>span:first-child {{
            border-radius: 4px !important;
        }}

        /* File uploader */
        div[data-testid="stFileUploader"] {{
            border: 2px dashed var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            background: var(--bg-secondary) !important;
        }}

        div[data-testid="stFileUploader"]:hover {{
            border-color: var(--primary) !important;
            background: var(--primary-light) !important;
        }}

        /* Selectbox dropdown */
        div[data-baseweb="select"] > div {{
            background: var(--card-bg) !important;
            border-color: var(--border-color) !important;
        }}

        div[data-baseweb="popover"] {{
            background: var(--card-bg) !important;
        }}

        div[data-baseweb="menu"] {{
            background: var(--card-bg) !important;
        }}

        li[role="option"] {{
            color: var(--text-primary) !important;
        }}

        li[role="option"]:hover {{
            background: var(--primary-light) !important;
        }}

        /* Expander header text */
        div[data-testid="stExpander"] summary {{
            color: var(--text-primary) !important;
        }}

        div[data-testid="stExpander"] summary:hover {{
            color: var(--primary) !important;
        }}

        /* Step indicator styles */
        .step-indicator {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 2rem;
        }}

        .step {{
            display: flex;
            flex-direction: column;
            align-items: center;
            flex: 1;
        }}

        .step-number {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}

        .step-number.completed {{
            background: var(--success);
            color: white;
        }}

        .step-number.active {{
            background: var(--primary);
            color: white;
        }}

        .step-number.pending {{
            background: var(--bg-tertiary);
            color: var(--text-muted);
        }}

        /* Card component */
        .custom-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            margin-bottom: 1rem;
        }}

        .custom-card:hover {{
            box-shadow: var(--shadow-md);
            border-color: var(--primary);
        }}

        .custom-card-header {{
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.75rem;
        }}

        .custom-card-body {{
            color: var(--text-secondary);
            line-height: 1.6;
        }}

        /* Badge styles */
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }}

        .badge-primary {{
            background: var(--primary-light);
            color: var(--primary);
        }}

        .badge-success {{
            background: rgba(34, 197, 94, 0.15);
            color: {COLORS['success-500']};
        }}

        .badge-warning {{
            background: rgba(245, 158, 11, 0.15);
            color: {COLORS['warning-500']};
        }}

        .badge-error {{
            background: rgba(239, 68, 68, 0.15);
            color: {COLORS['error-500']};
        }}

        /* Animated gradient background for headers */
        .gradient-header {{
            background: linear-gradient(135deg, var(--primary) 0%, {COLORS['primary-700']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* Model cards */
        .model-card {{
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: 1.25rem;
            cursor: pointer;
        }}

        .model-card:hover {{
            border-color: var(--primary);
            box-shadow: var(--shadow-md);
        }}

        .model-card.selected {{
            border-color: var(--primary);
            background: var(--primary-light);
        }}

        .model-card-title {{
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }}

        .model-card-subtitle {{
            font-size: 0.875rem;
            color: var(--text-muted);
        }}

        /* Tooltip styles */
        .tooltip {{
            position: relative;
        }}

        .tooltip .tooltip-text {{
            visibility: hidden;
            background-color: var(--bg-tertiary);
            color: var(--text-primary);
            padding: 0.5rem 0.75rem;
            border-radius: var(--border-radius);
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.75rem;
            white-space: nowrap;
            border: 1px solid var(--border-color);
        }}

        .tooltip:hover .tooltip-text {{
            visibility: visible;
        }}

        /* Loading animation */
        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.5;
            }}
        }}

        .loading {{
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }}

        /* Fade in animation */
        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .fade-in {{
            animation: fadeIn 0.3s ease-out;
        }}

        /* ============================================
           HELP SYSTEM - Footer Display
           ============================================ */
        
        /* Help footer container - fixed at bottom */
        .help-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, var(--bg-secondary) 0%, rgba(26, 29, 36, 0.95) 100%);
            border-top: 1px solid var(--border-color);
            padding: 0.5rem 1.5rem 0.5rem calc(1rem + 245px);
            z-index: 999;
            min-height: 38px;
            display: flex;
            align-items: center;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: all 0.2s ease-in-out;
        }}
        
        /* Adjust for collapsed sidebar */
        [data-testid="stSidebar"][aria-expanded="false"] ~ .main .help-footer {{
            padding-left: 1.5rem;
        }}
        
        .help-footer.has-content {{
            padding-top: 0.6rem;
            padding-bottom: 0.6rem;
        }}
        
        /* Help content styling */
        .help-content {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            font-size: 0.8rem;
            line-height: 1.5;
            animation: fadeInHelp 0.15s ease-out;
            max-width: 100%;
        }}
        
        @keyframes fadeInHelp {{
            from {{ opacity: 0; transform: translateY(3px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .help-icon {{
            font-size: 0.9rem;
            opacity: 0.7;
            flex-shrink: 0;
        }}
        
        .help-title {{
            font-weight: 600;
            color: var(--primary);
            flex-shrink: 0;
        }}
        
        .help-separator {{
            color: var(--text-muted);
            opacity: 0.5;
            flex-shrink: 0;
        }}
        
        .help-short {{
            color: var(--text-primary);
        }}
        
        .help-detail {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            opacity: 0.85;
        }}
        
        .help-tip {{
            color: {COLORS['warning-500']};
            font-size: 0.75rem;
            font-style: italic;
            flex-shrink: 0;
        }}
        
        /* Placeholder text when no help */
        .help-placeholder {{
            color: var(--text-muted);
            font-size: 0.75rem;
            font-style: italic;
            opacity: 0.5;
        }}
        
        /* Add bottom padding to main content to prevent overlap */
        .main .block-container {{
            padding-bottom: 4rem !important;
        }}
        
        /* Help indicator icon - subtle question mark */
        .help-indicator {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 14px;
            height: 14px;
            font-size: 9px;
            font-weight: 600;
            color: var(--text-muted);
            border: 1px solid var(--border-color);
            border-radius: 50%;
            margin-left: 4px;
            opacity: 0.4;
            cursor: help;
            transition: all 0.2s;
            vertical-align: middle;
        }}
        
        .help-indicator:hover {{
            opacity: 1;
            border-color: var(--primary);
            color: var(--primary);
            background: var(--primary-light);
        }}
        
        /* Style for labels with help */
        .label-with-help {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            cursor: help;
        }}
        
        /* Hoverable elements with subtle underline */
        [data-help] {{
            cursor: help;
            border-bottom: 1px dotted var(--border-color);
            transition: border-color 0.2s;
        }}
        
        [data-help]:hover {{
            border-bottom-color: var(--primary);
        }}

    </style>
    """


def render_card(title: str, content: str, icon: str = None) -> str:
    """Render a styled card component."""
    icon_html = f'<span style="margin-right: 0.5rem;">{icon}</span>' if icon else ''
    return f"""
    <div class="custom-card fade-in">
        <div class="custom-card-header">{icon_html}{title}</div>
        <div class="custom-card-body">{content}</div>
    </div>
    """


def render_badge(text: str, variant: str = "primary") -> str:
    """Render a styled badge.
    
    Args:
        text: Badge text
        variant: One of 'primary', 'success', 'warning', 'error'
    """
    return f'<span class="badge badge-{variant}">{text}</span>'


def render_step_indicator(steps: list, current_step: int) -> str:
    """Render a step indicator for wizard-like flows.
    
    Args:
        steps: List of step names
        current_step: Current step index (0-based)
    """
    step_html = []
    for i, step_name in enumerate(steps):
        if i < current_step:
            status = "completed"
            icon = "✓"
        elif i == current_step:
            status = "active"
            icon = str(i + 1)
        else:
            status = "pending"
            icon = str(i + 1)
        
        step_html.append(f"""
        <div class="step">
            <div class="step-number {status}">{icon}</div>
            <span style="font-size: 0.875rem; color: {'var(--gray-800)' if status != 'pending' else 'var(--gray-400)'};">
                {step_name}
            </span>
        </div>
        """)
    
    return f'<div class="step-indicator">{"".join(step_html)}</div>'


def render_model_card(
    name: str,
    size: str,
    vram: str,
    description: str,
    recommended: bool = False,
    selected: bool = False
) -> str:
    """Render a model selection card."""
    selected_class = "selected" if selected else ""
    badge = render_badge("Recommended", "success") if recommended else ""
    
    return f"""
    <div class="model-card {selected_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div class="model-card-title">{name}</div>
                <div class="model-card-subtitle">{description}</div>
            </div>
            {badge}
        </div>
        <div style="margin-top: 0.75rem; display: flex; gap: 1rem;">
            <span class="badge badge-primary">Size: {size}</span>
            <span class="badge badge-warning">VRAM: {vram}</span>
        </div>
    </div>
    """


def render_stat_card(
    title: str,
    value: str,
    subtitle: str = None,
    icon: str = None,
    color: str = "primary"
) -> str:
    """Render a statistics card."""
    icon_html = f'<span style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</span>' if icon else ''
    subtitle_html = f'<div style="font-size: 0.75rem; color: var(--gray-500);">{subtitle}</div>' if subtitle else ''
    
    return f"""
    <div class="custom-card" style="text-align: center;">
        {icon_html}
        <div style="font-size: 2rem; font-weight: 700; color: var(--{color});">{value}</div>
        <div style="font-size: 0.875rem; color: var(--gray-600); margin-top: 0.25rem;">{title}</div>
        {subtitle_html}
    </div>
    """


def render_progress_steps(current: int, total: int, label: str = "Progress") -> str:
    """Render a visual progress indicator."""
    percentage = (current / total) * 100 if total > 0 else 0
    
    return f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="font-size: 0.875rem; font-weight: 500; color: var(--gray-700);">{label}</span>
            <span style="font-size: 0.875rem; color: var(--gray-500);">{current}/{total}</span>
        </div>
        <div style="height: 8px; background: var(--gray-200); border-radius: 9999px; overflow: hidden;">
            <div style="height: 100%; width: {percentage}%; background: linear-gradient(90deg, var(--primary) 0%, {COLORS['primary-400']} 100%); border-radius: 9999px;"></div>
        </div>
    </div>
    """


def apply_theme():
    """Apply the custom theme to the Streamlit app.
    Call this at the beginning of your main app file.
    """
    # Page configuration should already be set in main.py
    
    # Inject custom CSS
    inject_custom_css()
    
    # Add any additional theme configuration here
    pass


# Utility function to create consistent spacing
def spacer(size: str = "md"):
    """Add vertical spacing.
    
    Args:
        size: One of 'sm' (1rem), 'md' (2rem), 'lg' (3rem), 'xl' (4rem)
    """
    sizes = {"sm": "1rem", "md": "2rem", "lg": "3rem", "xl": "4rem"}
    st.markdown(f'<div style="height: {sizes.get(size, "2rem")};"></div>', unsafe_allow_html=True)
