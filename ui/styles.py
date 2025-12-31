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
           ============================================ */
        
        /* Root variables */
        :root {{
            --primary: {COLORS['primary-500']};
            --primary-hover: {COLORS['primary-600']};
            --primary-light: {COLORS['primary-100']};
            --success: {COLORS['success-500']};
            --warning: {COLORS['warning-500']};
            --error: {COLORS['error-500']};
            --gray-50: {COLORS['gray-50']};
            --gray-100: {COLORS['gray-100']};
            --gray-200: {COLORS['gray-200']};
            --gray-700: {COLORS['gray-700']};
            --gray-800: {COLORS['gray-800']};
            --gray-900: {COLORS['gray-900']};
            --border-radius: 0.5rem;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }}

        /* Smooth transitions everywhere */
        * {{
            transition: all 0.2s ease-in-out;
        }}

        /* Main container */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}

        /* Headers */
        h1 {{
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            color: var(--gray-900) !important;
        }}

        h2 {{
            font-weight: 600 !important;
            color: var(--gray-800) !important;
        }}

        h3 {{
            font-weight: 600 !important;
            color: var(--gray-700) !important;
        }}

        /* Cards/Containers */
        div[data-testid="stExpander"] {{
            border: 1px solid var(--gray-200) !important;
            border-radius: var(--border-radius) !important;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }}

        div[data-testid="stExpander"]:hover {{
            box-shadow: var(--shadow-md);
            border-color: var(--primary) !important;
        }}

        /* Primary buttons */
        .stButton>button[kind="primary"] {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%) !important;
            border: none !important;
            border-radius: var(--border-radius) !important;
            padding: 0.625rem 1.25rem !important;
            font-weight: 600 !important;
            box-shadow: var(--shadow-md);
        }}

        .stButton>button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: var(--shadow-lg);
        }}

        /* Secondary buttons */
        .stButton>button[kind="secondary"] {{
            background: white !important;
            border: 1px solid var(--gray-300) !important;
            border-radius: var(--border-radius) !important;
            color: var(--gray-700) !important;
            font-weight: 500 !important;
        }}

        .stButton>button[kind="secondary"]:hover {{
            background: var(--gray-50) !important;
            border-color: var(--primary) !important;
            color: var(--primary) !important;
        }}

        /* Input fields */
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>div {{
            border-radius: var(--border-radius) !important;
            border: 1px solid var(--gray-300) !important;
        }}

        .stTextInput>div>div>input:focus,
        .stTextArea>div>div>textarea:focus {{
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px var(--primary-light) !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            background: transparent;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: var(--border-radius) var(--border-radius) 0 0;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            background: var(--gray-100);
            border: 1px solid var(--gray-200);
            border-bottom: none;
        }}

        .stTabs [aria-selected="true"] {{
            background: white !important;
            border-color: var(--primary) !important;
            border-bottom: 2px solid white !important;
            color: var(--primary) !important;
        }}

        /* Metrics */
        div[data-testid="stMetric"] {{
            background: linear-gradient(135deg, var(--gray-50) 0%, white 100%);
            border: 1px solid var(--gray-200);
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

        /* Progress bar */
        .stProgress>div>div>div>div {{
            background: linear-gradient(90deg, var(--primary) 0%, {COLORS['primary-400']} 100%) !important;
            border-radius: 9999px;
        }}

        /* Success/Info/Warning/Error messages */
        .stSuccess {{
            background: {COLORS['success-50']} !important;
            border-left: 4px solid var(--success) !important;
            border-radius: var(--border-radius) !important;
        }}

        .stInfo {{
            background: var(--primary-light) !important;
            border-left: 4px solid var(--primary) !important;
            border-radius: var(--border-radius) !important;
        }}

        .stWarning {{
            background: {COLORS['warning-50']} !important;
            border-left: 4px solid var(--warning) !important;
            border-radius: var(--border-radius) !important;
        }}

        .stError {{
            background: {COLORS['error-50']} !important;
            border-left: 4px solid var(--error) !important;
            border-radius: var(--border-radius) !important;
        }}

        /* Dataframe styling */
        div[data-testid="stDataFrame"] {{
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--gray-50) 0%, white 100%);
            border-right: 1px solid var(--gray-200);
        }}

        section[data-testid="stSidebar"] .stButton>button {{
            width: 100%;
            justify-content: flex-start;
            padding-left: 1rem;
        }}

        /* Radio buttons */
        .stRadio>div {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .stRadio>div>label {{
            background: white;
            border: 1px solid var(--gray-300);
            border-radius: var(--border-radius);
            padding: 0.5rem 1rem;
            cursor: pointer;
        }}

        .stRadio>div>label:hover {{
            border-color: var(--primary);
            background: var(--primary-light);
        }}

        /* Divider */
        hr {{
            border: none;
            border-top: 1px solid var(--gray-200);
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
            background: var(--gray-200);
            color: var(--gray-500);
        }}

        /* Card component */
        .custom-card {{
            background: white;
            border: 1px solid var(--gray-200);
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
            color: var(--gray-800);
            margin-bottom: 0.75rem;
        }}

        .custom-card-body {{
            color: var(--gray-600);
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
            background: {COLORS['success-50']};
            color: {COLORS['success-600']};
        }}

        .badge-warning {{
            background: {COLORS['warning-50']};
            color: {COLORS['warning-600']};
        }}

        .badge-error {{
            background: {COLORS['error-50']};
            color: {COLORS['error-600']};
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
            background: white;
            border: 2px solid var(--gray-200);
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
            color: var(--gray-800);
            margin-bottom: 0.25rem;
        }}

        .model-card-subtitle {{
            font-size: 0.875rem;
            color: var(--gray-500);
        }}

        /* Tooltip styles */
        .tooltip {{
            position: relative;
        }}

        .tooltip .tooltip-text {{
            visibility: hidden;
            background-color: var(--gray-800);
            color: white;
            padding: 0.5rem 0.75rem;
            border-radius: var(--border-radius);
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.75rem;
            white-space: nowrap;
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
