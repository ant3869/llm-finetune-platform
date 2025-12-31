"""
Batch Inference and Evaluation System.

Provides:
- Batch processing of test cases
- Automated evaluation metrics (BLEU, ROUGE, semantic similarity)
- Comparison between models
- Results export
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single example."""
    input_text: str
    expected_output: str
    generated_output: str
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    tokens_generated: int = 0


@dataclass
class BatchResult:
    """Results from a batch evaluation run."""
    model_name: str
    timestamp: str
    total_samples: int
    results: List[EvaluationResult]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "total_samples": self.total_samples,
            "aggregate_metrics": self.aggregate_metrics,
            "total_time_seconds": self.total_time_seconds,
            "results": [
                {
                    "input": r.input_text[:200] + "..." if len(r.input_text) > 200 else r.input_text,
                    "expected": r.expected_output[:200] + "..." if len(r.expected_output) > 200 else r.expected_output,
                    "generated": r.generated_output[:200] + "..." if len(r.generated_output) > 200 else r.generated_output,
                    "metrics": r.metrics,
                    "latency_ms": r.latency_ms,
                }
                for r in self.results
            ]
        }
    
    def save(self, output_path: str):
        """Save results to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {path}")


class MetricsCalculator:
    """Calculate evaluation metrics for generated text."""
    
    def __init__(self):
        self._bleu_available = False
        self._rouge_available = False
        
        # Try to import optional metrics libraries
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            self._bleu_available = True
            self._smoothing = SmoothingFunction()
        except ImportError:
            logger.warning("NLTK not available, BLEU scores disabled")
        
        try:
            from rouge_score import rouge_scorer
            self._rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self._rouge_available = True
        except ImportError:
            logger.warning("rouge-score not available, ROUGE scores disabled")
    
    def calculate_all(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate all available metrics."""
        metrics = {}
        
        # Basic metrics (always available)
        metrics["length_ratio"] = len(hypothesis) / max(len(reference), 1)
        metrics["word_overlap"] = self._word_overlap(reference, hypothesis)
        
        # BLEU score
        if self._bleu_available:
            try:
                metrics["bleu"] = self._calculate_bleu(reference, hypothesis)
            except Exception as e:
                logger.warning(f"BLEU calculation failed: {e}")
        
        # ROUGE scores
        if self._rouge_available:
            try:
                rouge_scores = self._calculate_rouge(reference, hypothesis)
                metrics.update(rouge_scores)
            except Exception as e:
                logger.warning(f"ROUGE calculation failed: {e}")
        
        return metrics
    
    def bleu_score(self, hypothesis: str, reference: str) -> float:
        """Calculate BLEU score (public method)."""
        if self._bleu_available:
            return self._calculate_bleu(reference, hypothesis)
        return 0.0
    
    def rouge_scores(self, hypothesis: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores (public method)."""
        if self._rouge_available:
            scores = self._calculate_rouge(reference, hypothesis)
            return {
                "rouge-1": scores.get("rouge1_f", 0),
                "rouge-l": scores.get("rougeL_f", 0),
            }
        return {"rouge-1": 0, "rouge-l": 0}
    
    def word_overlap(self, hypothesis: str, reference: str) -> float:
        """Calculate word overlap (public method)."""
        return self._word_overlap(reference, hypothesis)
    
    def _word_overlap(self, reference: str, hypothesis: str) -> float:
        """Calculate simple word overlap ratio."""
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        
        if not ref_words:
            return 0.0
        
        overlap = len(ref_words & hyp_words)
        return overlap / len(ref_words)
    
    def _calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score."""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if not hyp_tokens:
            return 0.0
        
        smoothing = SmoothingFunction()
        score = sentence_bleu(
            [ref_tokens], 
            hyp_tokens,
            smoothing_function=smoothing.method1
        )
        return score
    
    def _calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self._rouge_scorer.score(reference, hypothesis)
        
        return {
            "rouge1_f": scores["rouge1"].fmeasure,
            "rouge2_f": scores["rouge2"].fmeasure,
            "rougeL_f": scores["rougeL"].fmeasure,
        }


class BatchEvaluator:
    """
    Run batch inference and evaluation on test datasets.
    """
    
    def __init__(self, inference_engine=None):
        """
        Initialize batch evaluator.
        
        Args:
            inference_engine: InferenceEngine instance with loaded model
        """
        self.engine = inference_engine
        self.metrics_calculator = MetricsCalculator()
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None
    
    def set_engine(self, engine):
        """Set the inference engine."""
        self.engine = engine
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates: callback(current, total, status)"""
        self.progress_callback = callback
    
    def _notify_progress(self, current: int, total: int, status: str):
        """Notify progress callback if set."""
        if self.progress_callback:
            self.progress_callback(current, total, status)
    
    def load_test_data(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load test data from file.
        
        Expected format (JSON/JSONL):
        {"instruction": "...", "input": "...", "output": "..."}
        
        Returns list of test cases.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Test file not found: {file_path}")
        
        test_cases = []
        
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        test_cases.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    test_cases = data
                else:
                    test_cases = [data]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
        return test_cases
    
    def format_prompt(self, test_case: Dict[str, str]) -> str:
        """Format test case into prompt string."""
        instruction = test_case.get("instruction", "")
        input_text = test_case.get("input", "")
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        return prompt
    
    def run_evaluation(
        self,
        test_cases: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        model_name: str = "unknown",
    ) -> BatchResult:
        """
        Run batch evaluation on test cases.
        
        Args:
            test_cases: List of test cases with instruction/input/output
            max_new_tokens: Max tokens to generate per response
            temperature: Sampling temperature
            model_name: Name for identifying this model in results
            
        Returns:
            BatchResult with all evaluation results
        """
        if self.engine is None:
            raise RuntimeError("No inference engine set. Call set_engine() first.")
        
        results = []
        start_time = time.time()
        
        self._notify_progress(0, len(test_cases), "Starting evaluation...")
        
        for i, test_case in enumerate(test_cases):
            # Format prompt
            prompt = self.format_prompt(test_case)
            expected = test_case.get("output", "")
            
            # Generate response
            gen_start = time.time()
            try:
                generated = self.engine.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                logger.error(f"Generation failed for case {i}: {e}")
                generated = f"[ERROR: {e}]"
            
            gen_time = (time.time() - gen_start) * 1000  # ms
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all(expected, generated)
            
            # Create result
            result = EvaluationResult(
                input_text=prompt,
                expected_output=expected,
                generated_output=generated,
                metrics=metrics,
                latency_ms=gen_time,
                tokens_generated=len(generated.split()),  # Approximate
            )
            results.append(result)
            
            self._notify_progress(i + 1, len(test_cases), f"Processed {i + 1}/{len(test_cases)}")
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate_metrics(results)
        
        batch_result = BatchResult(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            total_samples=len(test_cases),
            results=results,
            aggregate_metrics=aggregate,
            total_time_seconds=total_time,
        )
        
        self._notify_progress(len(test_cases), len(test_cases), "Evaluation complete!")
        
        return batch_result
    
    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        if not results:
            return {}
        
        aggregate = {}
        
        # Collect all metric names
        metric_names = set()
        for r in results:
            metric_names.update(r.metrics.keys())
        
        # Calculate averages
        for metric_name in metric_names:
            values = [r.metrics.get(metric_name, 0) for r in results]
            aggregate[f"avg_{metric_name}"] = sum(values) / len(values)
        
        # Add latency stats
        latencies = [r.latency_ms for r in results]
        aggregate["avg_latency_ms"] = sum(latencies) / len(latencies)
        aggregate["min_latency_ms"] = min(latencies)
        aggregate["max_latency_ms"] = max(latencies)
        
        # Throughput
        total_tokens = sum(r.tokens_generated for r in results)
        total_time = sum(r.latency_ms for r in results) / 1000
        if total_time > 0:
            aggregate["tokens_per_second"] = total_tokens / total_time
        
        return aggregate
    
    def compare_models(
        self,
        results_list: List[BatchResult],
    ) -> Dict[str, Any]:
        """
        Compare results from multiple models.
        
        Args:
            results_list: List of BatchResult from different models
            
        Returns:
            Comparison summary
        """
        comparison = {
            "models": [],
            "metrics_comparison": {},
            "winner_by_metric": {},
        }
        
        for result in results_list:
            comparison["models"].append({
                "name": result.model_name,
                "samples": result.total_samples,
                "total_time": result.total_time_seconds,
                "metrics": result.aggregate_metrics,
            })
        
        # Find best model for each metric
        if results_list:
            metric_names = set()
            for r in results_list:
                metric_names.update(r.aggregate_metrics.keys())
            
            for metric in metric_names:
                best_model = None
                best_value = None
                
                for result in results_list:
                    value = result.aggregate_metrics.get(metric, 0)
                    
                    # Higher is better for most metrics (except latency)
                    is_better = (
                        best_value is None or
                        ("latency" in metric and value < best_value) or
                        ("latency" not in metric and value > best_value)
                    )
                    
                    if is_better:
                        best_value = value
                        best_model = result.model_name
                
                comparison["winner_by_metric"][metric] = {
                    "model": best_model,
                    "value": best_value,
                }
        
        return comparison
