"""
Progress Tracker Component

Real-time training progress visualization with:
- Progress bar
- Loss chart
- VRAM usage
- Training metrics
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class TrainingMetrics:
    """Container for training metrics history."""
    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    vram_usage: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    
    def add_point(self, step: int, loss: float, lr: float, vram: float):
        """Add a data point."""
        self.steps.append(step)
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.vram_usage.append(vram)
        self.timestamps.append(datetime.now())
    
    def clear(self):
        """Clear all metrics."""
        self.steps.clear()
        self.losses.clear()
        self.learning_rates.clear()
        self.vram_usage.clear()
        self.timestamps.clear()


def render_progress_header(progress: Dict[str, Any]):
    """Render the progress header with key metrics."""
    is_gpu_mode = progress.get('is_gpu_mode', True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Cap progress at 100% for display
    progress_pct = min(100.0, progress.get('progress_percent', 0))
    
    with col1:
        st.metric(
            "Progress",
            f"{progress_pct:.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Step",
            f"{progress.get('current_step', 0)}/{progress.get('total_steps', 0)}"
        )
    
    with col3:
        st.metric(
            "Epoch",
            f"{progress.get('current_epoch', 0):.1f}/{progress.get('total_epochs', 0)}"
        )
    
    with col4:
        st.metric(
            "Loss",
            f"{progress.get('loss', 0):.4f}"
        )
    
    with col5:
        if is_gpu_mode:
            vram_used = progress.get('vram_used_gb', 0)
            vram_total = progress.get('vram_total_gb', 8)
            st.metric(
                "VRAM",
                f"{vram_used:.1f}/{vram_total:.1f} GB"
            )
        else:
            ram_used = progress.get('ram_used_gb', progress.get('vram_used_gb', 0))
            ram_total = progress.get('ram_total_gb', progress.get('vram_total_gb', 16))
            cpu_pct = progress.get('cpu_percent', 0)
            st.metric(
                "RAM",
                f"{ram_used:.1f}/{ram_total:.1f} GB",
                delta=f"CPU: {cpu_pct:.0f}%"
            )


def render_progress_bar(progress: Dict[str, Any]):
    """Render the main progress bar."""
    pct = progress.get('progress_percent', 0) / 100
    # Clamp to valid range [0.0, 1.0] for Streamlit progress bar
    pct = max(0.0, min(1.0, pct))
    
    st.progress(pct)
    
    # Time estimates
    elapsed = progress.get('elapsed_time', 0)
    remaining = progress.get('estimated_remaining', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        elapsed_str = format_time(elapsed)
        st.caption(f"⏱️ Elapsed: {elapsed_str}")
    
    with col2:
        remaining_str = format_time(remaining)
        st.caption(f"⏳ Remaining: {remaining_str}")
    
    with col3:
        samples_sec = progress.get('samples_per_second', 0)
        st.caption(f"⚡ Speed: {samples_sec:.2f} steps/sec")


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds <= 0:
        return "--:--"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def render_loss_chart(metrics: TrainingMetrics, is_gpu_mode: bool = True):
    """Render the loss chart."""
    if not metrics.steps:
        st.info("Loss chart will appear when training starts...")
        return
    
    memory_label = "VRAM Usage" if is_gpu_mode else "RAM Usage"
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training Loss", memory_label),
        horizontal_spacing=0.1
    )
    
    # Loss curve
    fig.add_trace(
        go.Scatter(
            x=metrics.steps,
            y=metrics.losses,
            mode='lines+markers',
            name='Loss',
            line=dict(color='#4CAF50', width=2),
            marker=dict(size=4),
        ),
        row=1, col=1
    )
    
    # Memory usage (VRAM or RAM)
    memory_name = 'VRAM (GB)' if is_gpu_mode else 'RAM (GB)'
    fig.add_trace(
        go.Scatter(
            x=metrics.steps,
            y=metrics.vram_usage,
            mode='lines+markers',
            name=memory_name,
            line=dict(color='#2196F3', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)',
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="GB", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def render_training_status(status: str):
    """Render training status indicator."""
    status_config = {
        "idle": ("⚪", "Ready to start", "info"),
        "loading": ("🔄", "Loading model...", "info"),
        "training": ("🟢", "Training in progress", "success"),
        "paused": ("🟡", "Training paused", "warning"),
        "stopping": ("🟠", "Stopping...", "warning"),
        "completed": ("✅", "Training complete!", "success"),
        "error": ("🔴", "Error occurred", "error"),
    }
    
    icon, message, type_ = status_config.get(status, ("⚪", status, "info"))
    
    if type_ == "success":
        st.success(f"{icon} {message}")
    elif type_ == "warning":
        st.warning(f"{icon} {message}")
    elif type_ == "error":
        st.error(f"{icon} {message}")
    else:
        st.info(f"{icon} {message}")


def render_vram_gauge(used: float, total: float):
    """Render a VRAM usage gauge."""
    pct = (used / total * 100) if total > 0 else 0
    
    # Determine color based on usage
    if pct < 70:
        color = "#4CAF50"  # Green
    elif pct < 85:
        color = "#FF9800"  # Orange
    else:
        color = "#F44336"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=used,
        number={'suffix': " GB", 'font': {'size': 24}},
        delta={'reference': total * 0.8, 'relative': False, 'position': "bottom"},
        title={'text': "VRAM Usage", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, total], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, total * 0.7], 'color': 'rgba(76, 175, 80, 0.1)'},
                {'range': [total * 0.7, total * 0.85], 'color': 'rgba(255, 152, 0, 0.1)'},
                {'range': [total * 0.85, total], 'color': 'rgba(244, 67, 54, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': total * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_training_log(log_entries: List[str], max_entries: int = 20):
    """Render a scrollable training log."""
    st.markdown("#### 📜 Training Log")
    
    if not log_entries:
        st.caption("Log entries will appear here...")
        return
    
    # Show last N entries
    recent = log_entries[-max_entries:]
    
    log_text = "\n".join(recent)
    
    st.code(log_text, language="text")


class ProgressTracker:
    """
    Progress tracker that manages UI updates during training.
    
    Usage:
        tracker = ProgressTracker()
        trainer.add_progress_callback(tracker.update)
        # Training runs...
        tracker.render()  # In Streamlit
    """
    
    def __init__(self):
        self.metrics = TrainingMetrics()
        self.current_progress = {}
        self.log_entries = []
        self.status = "idle"
        self.is_gpu_mode = True
    
    def update(self, progress):
        """Callback to receive progress updates from trainer."""
        # Convert to dict if it's a TrainingProgress object
        if hasattr(progress, 'to_dict'):
            progress = progress.to_dict()
        
        self.current_progress = progress
        self.status = progress.get('status', 'training')
        self.is_gpu_mode = progress.get('is_gpu_mode', True)
        
        # Add to metrics history
        step = progress.get('current_step', 0)
        loss = progress.get('loss', 0)
        lr = progress.get('learning_rate', 0)
        # Use RAM for CPU mode, VRAM for GPU mode
        memory = progress.get('vram_used_gb', 0)
        
        if step > 0 and (not self.metrics.steps or step > self.metrics.steps[-1]):
            self.metrics.add_point(step, loss, lr, memory)
            
            # Add log entry with appropriate label
            memory_label = "VRAM" if self.is_gpu_mode else "RAM"
            self.log_entries.append(
                f"[Step {step}] Loss: {loss:.4f} | LR: {lr:.2e} | {memory_label}: {memory:.1f}GB"
            )
    
    def render(self):
        """Render all progress components."""
        render_training_status(self.status)
        render_progress_header(self.current_progress)
        render_progress_bar(self.current_progress)
        render_loss_chart(self.metrics, self.is_gpu_mode)
    
    def reset(self):
        """Reset the tracker for a new training run."""
        self.metrics.clear()
        self.current_progress = {}
        self.log_entries = []
        self.status = "idle"
