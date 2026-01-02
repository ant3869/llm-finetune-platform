"""
Hyperparameter Optimization Module - Milestone 5

Provides automated hyperparameter search for fine-tuning:
- Grid Search: Exhaustive search over parameter grid
- Random Search: Random sampling of parameter space
- Bayesian Optimization: Smart search using past results (optional)

Optimized for memory-constrained environments (8GB VRAM).
"""

import logging
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
import math

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Define the search space for hyperparameters."""
    
    # Learning rate options
    learning_rates: List[float] = field(default_factory=lambda: [1e-5, 2e-5, 5e-5, 1e-4, 2e-4])
    
    # LoRA configuration
    lora_r_values: List[int] = field(default_factory=lambda: [8, 16, 32])
    lora_alpha_values: List[int] = field(default_factory=lambda: [16, 32, 64])
    lora_dropout_values: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    
    # Training configuration
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2])  # Limited for 8GB VRAM
    gradient_accumulation_steps: List[int] = field(default_factory=lambda: [8, 16, 32])
    warmup_ratios: List[float] = field(default_factory=lambda: [0.0, 0.03, 0.1])
    
    # Sequence length (memory critical)
    max_seq_lengths: List[int] = field(default_factory=lambda: [256, 512])
    
    # Epochs
    epochs: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    def get_total_combinations(self) -> int:
        """Calculate total number of possible combinations."""
        return (
            len(self.learning_rates) *
            len(self.lora_r_values) *
            len(self.lora_alpha_values) *
            len(self.lora_dropout_values) *
            len(self.batch_sizes) *
            len(self.gradient_accumulation_steps) *
            len(self.warmup_ratios) *
            len(self.max_seq_lengths) *
            len(self.epochs)
        )
    
    def generate_grid(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        combinations = []
        
        for params in product(
            self.learning_rates,
            self.lora_r_values,
            self.lora_alpha_values,
            self.lora_dropout_values,
            self.batch_sizes,
            self.gradient_accumulation_steps,
            self.warmup_ratios,
            self.max_seq_lengths,
            self.epochs,
        ):
            combinations.append({
                "learning_rate": params[0],
                "lora_r": params[1],
                "lora_alpha": params[2],
                "lora_dropout": params[3],
                "batch_size": params[4],
                "gradient_accumulation_steps": params[5],
                "warmup_ratio": params[6],
                "max_seq_length": params[7],
                "epochs": params[8],
            })
        
        return combinations
    
    def sample_random(self, n_samples: int = 10) -> List[Dict[str, Any]]:
        """Generate random samples from the search space."""
        samples = []
        
        for _ in range(n_samples):
            samples.append({
                "learning_rate": random.choice(self.learning_rates),
                "lora_r": random.choice(self.lora_r_values),
                "lora_alpha": random.choice(self.lora_alpha_values),
                "lora_dropout": random.choice(self.lora_dropout_values),
                "batch_size": random.choice(self.batch_sizes),
                "gradient_accumulation_steps": random.choice(self.gradient_accumulation_steps),
                "warmup_ratio": random.choice(self.warmup_ratios),
                "max_seq_length": random.choice(self.max_seq_lengths),
                "epochs": random.choice(self.epochs),
            })
        
        return samples
    
    @classmethod
    def quick_search(cls) -> "HyperparameterSpace":
        """Minimal search space for quick testing."""
        return cls(
            learning_rates=[1e-4, 2e-4],
            lora_r_values=[16],
            lora_alpha_values=[32],
            lora_dropout_values=[0.05],
            batch_sizes=[1],
            gradient_accumulation_steps=[16],
            warmup_ratios=[0.03],
            max_seq_lengths=[256],
            epochs=[1],
        )
    
    @classmethod
    def balanced_search(cls) -> "HyperparameterSpace":
        """Balanced search space for practical optimization."""
        return cls(
            learning_rates=[5e-5, 1e-4, 2e-4],
            lora_r_values=[8, 16, 32],
            lora_alpha_values=[16, 32],
            lora_dropout_values=[0.05],
            batch_sizes=[1],
            gradient_accumulation_steps=[16],
            warmup_ratios=[0.03],
            max_seq_lengths=[256, 512],
            epochs=[3],
        )
    
    @classmethod
    def thorough_search(cls) -> "HyperparameterSpace":
        """Comprehensive search space for thorough optimization."""
        return cls(
            learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
            lora_r_values=[8, 16, 32, 64],
            lora_alpha_values=[16, 32, 64],
            lora_dropout_values=[0.0, 0.05, 0.1],
            batch_sizes=[1],
            gradient_accumulation_steps=[8, 16, 32],
            warmup_ratios=[0.0, 0.03, 0.1],
            max_seq_lengths=[256, 512, 1024],
            epochs=[3, 5],
        )


@dataclass
class TrialResult:
    """Result from a single hyperparameter trial."""
    trial_id: int
    hyperparameters: Dict[str, Any]
    final_loss: float = float('inf')
    eval_loss: Optional[float] = None
    training_time_seconds: float = 0.0
    vram_peak_gb: float = 0.0
    status: str = "pending"  # pending, running, completed, failed, skipped
    error_message: Optional[str] = None
    adapter_path: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "hyperparameters": self.hyperparameters,
            "final_loss": self.final_loss,
            "eval_loss": self.eval_loss,
            "training_time_seconds": self.training_time_seconds,
            "vram_peak_gb": self.vram_peak_gb,
            "status": self.status,
            "error_message": self.error_message,
            "adapter_path": self.adapter_path,
            "timestamp": self.timestamp,
        }


@dataclass
class OptimizationResult:
    """Results from a complete optimization run."""
    search_method: str
    total_trials: int
    completed_trials: int
    best_trial_id: int
    best_hyperparameters: Dict[str, Any]
    best_loss: float
    trials: List[TrialResult]
    total_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_method": self.search_method,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "best_trial_id": self.best_trial_id,
            "best_hyperparameters": self.best_hyperparameters,
            "best_loss": self.best_loss,
            "total_time_seconds": self.total_time_seconds,
            "timestamp": self.timestamp,
            "trials": [t.to_dict() for t in self.trials],
        }
    
    def save(self, output_path: str):
        """Save optimization results to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Optimization results saved to {path}")
    
    def get_top_n_trials(self, n: int = 5) -> List[TrialResult]:
        """Get the top N trials by loss."""
        completed = [t for t in self.trials if t.status == "completed"]
        return sorted(completed, key=lambda t: t.final_loss)[:n]


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization for LLM fine-tuning.
    
    Supports:
    - Grid Search: Test all combinations
    - Random Search: Sample random configurations
    - Smart Search: Focus on promising regions (simplified Bayesian-like)
    """
    
    def __init__(
        self,
        model_name: str,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./models/hpo_runs",
    ):
        """
        Initialize the optimizer.
        
        Args:
            model_name: HuggingFace model ID
            train_dataset: Training dataset (HuggingFace Dataset)
            eval_dataset: Optional evaluation dataset
            output_dir: Directory for saving trial results
        """
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        
        self.trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None
        
        # Callbacks
        self.progress_callback: Optional[Callable[[int, int, TrialResult], None]] = None
        self.trial_complete_callback: Optional[Callable[[TrialResult], None]] = None
        
        # Control
        self._stop_requested = False
        
    def set_progress_callback(self, callback: Callable[[int, int, TrialResult], None]):
        """Set callback for progress updates: callback(current_trial, total_trials, trial_result)"""
        self.progress_callback = callback
    
    def set_trial_complete_callback(self, callback: Callable[[TrialResult], None]):
        """Set callback when a trial completes."""
        self.trial_complete_callback = callback
    
    def stop(self):
        """Request stop of optimization."""
        self._stop_requested = True
    
    def grid_search(
        self,
        search_space: HyperparameterSpace,
        max_trials: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Run exhaustive grid search over hyperparameter space.
        
        Args:
            search_space: Hyperparameter search space
            max_trials: Maximum trials (for partial grid search)
        """
        combinations = search_space.generate_grid()
        
        if max_trials and max_trials < len(combinations):
            logger.info(f"Grid search limited to {max_trials}/{len(combinations)} combinations")
            combinations = random.sample(combinations, max_trials)
        
        return self._run_trials(combinations, "grid_search")
    
    def random_search(
        self,
        search_space: HyperparameterSpace,
        n_trials: int = 10,
    ) -> OptimizationResult:
        """
        Run random search over hyperparameter space.
        
        Args:
            search_space: Hyperparameter search space
            n_trials: Number of random trials
        """
        combinations = search_space.sample_random(n_trials)
        return self._run_trials(combinations, "random_search")
    
    def smart_search(
        self,
        search_space: HyperparameterSpace,
        n_trials: int = 10,
        exploration_ratio: float = 0.3,
    ) -> OptimizationResult:
        """
        Run smart search that focuses on promising regions.
        
        Uses a simplified approach:
        1. Start with random exploration
        2. Focus on variations of best-performing configs
        
        Args:
            search_space: Hyperparameter search space
            n_trials: Number of trials
            exploration_ratio: Fraction of trials for exploration (vs exploitation)
        """
        n_explore = max(2, int(n_trials * exploration_ratio))
        n_exploit = n_trials - n_explore
        
        logger.info(f"Smart search: {n_explore} exploration + {n_exploit} exploitation trials")
        
        # Phase 1: Exploration - random sampling
        explore_configs = search_space.sample_random(n_explore)
        
        all_configs = explore_configs.copy()
        
        # Run exploration phase
        for i, config in enumerate(explore_configs):
            if self._stop_requested:
                break
            
            trial = self._run_single_trial(i + 1, config)
            self.trials.append(trial)
            
            if self.progress_callback:
                self.progress_callback(i + 1, n_trials, trial)
        
        # Phase 2: Exploitation - variations of best configs
        if not self._stop_requested and n_exploit > 0:
            completed = [t for t in self.trials if t.status == "completed"]
            
            if completed:
                # Sort by loss and get top performers
                top_trials = sorted(completed, key=lambda t: t.final_loss)[:3]
                
                for j in range(n_exploit):
                    if self._stop_requested:
                        break
                    
                    # Pick a top performer and create variation
                    base_config = random.choice(top_trials).hyperparameters.copy()
                    varied_config = self._create_variation(base_config, search_space)
                    
                    trial = self._run_single_trial(len(self.trials) + 1, varied_config)
                    self.trials.append(trial)
                    
                    if self.progress_callback:
                        self.progress_callback(n_explore + j + 1, n_trials, trial)
        
        return self._compile_results("smart_search")
    
    def _create_variation(
        self,
        base_config: Dict[str, Any],
        search_space: HyperparameterSpace,
    ) -> Dict[str, Any]:
        """Create a variation of a base configuration."""
        config = base_config.copy()
        
        # Randomly modify 1-2 parameters
        params_to_vary = random.sample([
            ("learning_rate", search_space.learning_rates),
            ("lora_r", search_space.lora_r_values),
            ("lora_alpha", search_space.lora_alpha_values),
            ("warmup_ratio", search_space.warmup_ratios),
        ], k=random.randint(1, 2))
        
        for param_name, options in params_to_vary:
            current_value = config.get(param_name)
            # Try to pick a neighboring value
            if current_value in options:
                idx = options.index(current_value)
                # Move to neighbor with some probability of staying
                if random.random() > 0.3:
                    if random.random() > 0.5 and idx < len(options) - 1:
                        config[param_name] = options[idx + 1]
                    elif idx > 0:
                        config[param_name] = options[idx - 1]
            else:
                config[param_name] = random.choice(options)
        
        return config
    
    def _run_trials(
        self,
        configurations: List[Dict[str, Any]],
        search_method: str,
    ) -> OptimizationResult:
        """Run multiple trials with given configurations."""
        self.trials = []
        self._stop_requested = False
        start_time = time.time()
        
        for i, config in enumerate(configurations):
            if self._stop_requested:
                logger.info("Optimization stopped by user")
                break
            
            trial = self._run_single_trial(i + 1, config)
            self.trials.append(trial)
            
            if self.progress_callback:
                self.progress_callback(i + 1, len(configurations), trial)
            
            if self.trial_complete_callback:
                self.trial_complete_callback(trial)
        
        total_time = time.time() - start_time
        return self._compile_results(search_method, total_time)
    
    def _run_single_trial(
        self,
        trial_id: int,
        hyperparameters: Dict[str, Any],
    ) -> TrialResult:
        """Run a single training trial with given hyperparameters."""
        trial = TrialResult(
            trial_id=trial_id,
            hyperparameters=hyperparameters,
            status="running",
        )
        
        logger.info(f"Starting trial {trial_id}: {hyperparameters}")
        
        try:
            from core.trainer import Trainer, TrainingConfig
            import torch
            
            # Create training config from hyperparameters
            config = TrainingConfig(
                model_name_or_path=self.model_name,
                learning_rate=hyperparameters["learning_rate"],
                lora_r=hyperparameters["lora_r"],
                lora_alpha=hyperparameters["lora_alpha"],
                lora_dropout=hyperparameters.get("lora_dropout", 0.05),
                batch_size=hyperparameters["batch_size"],
                gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
                warmup_ratio=hyperparameters["warmup_ratio"],
                max_seq_length=hyperparameters["max_seq_length"],
                epochs=hyperparameters["epochs"],
                # Reduced logging for HPO
                logging_steps=50,
                save_steps=9999,  # Don't save intermediate checkpoints
                eval_steps=9999,
            )
            
            # Create trainer
            trainer = Trainer(config)
            
            # Prepare model
            trainer.prepare_model(self.model_name)
            
            # Track VRAM
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Create trial output directory
            trial_output = self.output_dir / f"trial_{trial_id:03d}"
            
            start_time = time.time()
            
            # Run training
            adapter_path = trainer.train(
                self.train_dataset,
                self.eval_dataset,
                output_dir=str(trial_output),
            )
            
            # Gather results
            trial.training_time_seconds = time.time() - start_time
            trial.final_loss = trainer.progress.loss
            trial.adapter_path = str(adapter_path)
            trial.status = "completed"
            
            if torch.cuda.is_available():
                trial.vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            
            logger.info(f"Trial {trial_id} completed: loss={trial.final_loss:.4f}")
            
            # Cleanup to free VRAM
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            trial.status = "failed"
            trial.error_message = str(e)
            logger.error(f"Trial {trial_id} failed: {e}")
            
            # Cleanup on error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        
        return trial
    
    def _compile_results(
        self,
        search_method: str,
        total_time: Optional[float] = None,
    ) -> OptimizationResult:
        """Compile trials into optimization result."""
        completed = [t for t in self.trials if t.status == "completed"]
        
        if completed:
            best = min(completed, key=lambda t: t.final_loss)
            self.best_trial = best
        else:
            best = TrialResult(trial_id=0, hyperparameters={})
        
        result = OptimizationResult(
            search_method=search_method,
            total_trials=len(self.trials),
            completed_trials=len(completed),
            best_trial_id=best.trial_id,
            best_hyperparameters=best.hyperparameters,
            best_loss=best.final_loss,
            trials=self.trials,
            total_time_seconds=total_time or sum(t.training_time_seconds for t in self.trials),
        )
        
        # Save results
        result.save(str(self.output_dir / "optimization_results.json"))
        
        return result
    
    def estimate_time(
        self,
        search_space: HyperparameterSpace,
        search_method: str,
        n_trials: Optional[int] = None,
        minutes_per_trial: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Estimate total optimization time.
        
        Args:
            search_space: The hyperparameter space
            search_method: 'grid', 'random', or 'smart'
            n_trials: Number of trials (for random/smart)
            minutes_per_trial: Estimated minutes per trial
        
        Returns:
            Dictionary with time estimates
        """
        if search_method == "grid":
            total_combinations = search_space.get_total_combinations()
            n_trials = n_trials or total_combinations
            n_trials = min(n_trials, total_combinations)
        else:
            n_trials = n_trials or 10
        
        total_minutes = n_trials * minutes_per_trial
        
        return {
            "total_trials": n_trials,
            "minutes_per_trial": minutes_per_trial,
            "total_minutes": total_minutes,
            "total_hours": total_minutes / 60,
            "formatted": self._format_duration(total_minutes * 60),
        }
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"


class HPOPresets:
    """Predefined hyperparameter optimization presets."""
    
    @staticmethod
    def quick_test() -> Tuple[HyperparameterSpace, Dict[str, Any]]:
        """Quick test preset - 2-4 trials, ~10-20 minutes."""
        space = HyperparameterSpace.quick_search()
        settings = {
            "search_method": "random",
            "n_trials": 2,
            "description": "Quick validation that HPO works",
        }
        return space, settings
    
    @staticmethod
    def learning_rate_sweep() -> Tuple[HyperparameterSpace, Dict[str, Any]]:
        """Focus on finding optimal learning rate."""
        space = HyperparameterSpace(
            learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            lora_r_values=[16],
            lora_alpha_values=[32],
            lora_dropout_values=[0.05],
            batch_sizes=[1],
            gradient_accumulation_steps=[16],
            warmup_ratios=[0.03],
            max_seq_lengths=[512],
            epochs=[3],
        )
        settings = {
            "search_method": "grid",
            "n_trials": 6,
            "description": "Find optimal learning rate",
        }
        return space, settings
    
    @staticmethod
    def lora_optimization() -> Tuple[HyperparameterSpace, Dict[str, Any]]:
        """Focus on LoRA hyperparameters."""
        space = HyperparameterSpace(
            learning_rates=[1e-4],
            lora_r_values=[4, 8, 16, 32, 64],
            lora_alpha_values=[8, 16, 32, 64],
            lora_dropout_values=[0.0, 0.05, 0.1],
            batch_sizes=[1],
            gradient_accumulation_steps=[16],
            warmup_ratios=[0.03],
            max_seq_lengths=[512],
            epochs=[3],
        )
        settings = {
            "search_method": "random",
            "n_trials": 12,
            "description": "Optimize LoRA configuration",
        }
        return space, settings
    
    @staticmethod
    def balanced_optimization() -> Tuple[HyperparameterSpace, Dict[str, Any]]:
        """Balanced search across key parameters."""
        space = HyperparameterSpace.balanced_search()
        settings = {
            "search_method": "smart",
            "n_trials": 10,
            "description": "Balanced optimization of key parameters",
        }
        return space, settings
    
    @staticmethod
    def full_optimization() -> Tuple[HyperparameterSpace, Dict[str, Any]]:
        """Comprehensive optimization - longer runtime."""
        space = HyperparameterSpace.thorough_search()
        settings = {
            "search_method": "smart",
            "n_trials": 20,
            "description": "Thorough optimization (longer runtime)",
        }
        return space, settings


def get_recommended_space_for_vram(vram_gb: float) -> HyperparameterSpace:
    """
    Get recommended search space based on available VRAM.
    
    Args:
        vram_gb: Available VRAM in GB
    
    Returns:
        Appropriate HyperparameterSpace
    """
    if vram_gb < 6:
        # Very limited VRAM - minimal configurations
        return HyperparameterSpace(
            learning_rates=[1e-4, 2e-4],
            lora_r_values=[8, 16],
            lora_alpha_values=[16, 32],
            lora_dropout_values=[0.05],
            batch_sizes=[1],
            gradient_accumulation_steps=[16, 32],
            warmup_ratios=[0.03],
            max_seq_lengths=[256],
            epochs=[1, 3],
        )
    elif vram_gb < 10:
        # Standard 8GB VRAM - balanced configurations
        return HyperparameterSpace.balanced_search()
    else:
        # 10GB+ VRAM - can try more configurations
        return HyperparameterSpace.thorough_search()
