"""
Post-Tuning Test Suite - Comprehensive Model Evaluation

Provides systematic testing and rating for fine-tuned models:
- Before/After comparison (base vs tuned)
- Response rating system with persistence
- Predefined test scenarios
- Effectiveness metrics dashboard
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single test case for evaluation."""
    id: str
    category: str
    name: str
    prompt: str
    expected_qualities: List[str] = field(default_factory=list)
    reference_output: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)


@dataclass
class TestResponse:
    """Response from a model for a test case."""
    test_id: str
    model_name: str
    model_type: str  # "base" or "tuned"
    adapter_name: Optional[str]
    response: str
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Ratings (1-5 scale)
    relevance_rating: Optional[int] = None
    accuracy_rating: Optional[int] = None
    quality_rating: Optional[int] = None
    helpfulness_rating: Optional[int] = None
    overall_rating: Optional[int] = None
    
    # Notes
    notes: str = ""
    
    def average_rating(self) -> Optional[float]:
        """Calculate average of all ratings."""
        ratings = [r for r in [
            self.relevance_rating,
            self.accuracy_rating,
            self.quality_rating,
            self.helpfulness_rating,
            self.overall_rating
        ] if r is not None]
        
        return sum(ratings) / len(ratings) if ratings else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "adapter_name": self.adapter_name,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "ratings": {
                "relevance": self.relevance_rating,
                "accuracy": self.accuracy_rating,
                "quality": self.quality_rating,
                "helpfulness": self.helpfulness_rating,
                "overall": self.overall_rating,
            },
            "average_rating": self.average_rating(),
            "notes": self.notes,
        }


@dataclass
class ComparisonResult:
    """Result comparing base vs tuned model."""
    test_case: TestCase
    base_response: TestResponse
    tuned_response: TestResponse
    winner: Optional[str] = None  # "base", "tuned", "tie"
    improvement_score: float = 0.0  # -1 to 1, positive means tuned is better


class TestSuiteManager:
    """Manage test suites and results."""
    
    def __init__(self, storage_path: str = "./logs/test_results"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.test_cases: Dict[str, TestCase] = {}
        self.responses: List[TestResponse] = []
        self.comparisons: List[ComparisonResult] = []
        
        # Load built-in test cases
        self._load_builtin_tests()
    
    def _load_builtin_tests(self):
        """Load predefined test cases for IT support."""
        builtin_tests = [
            # Ticket Summarization
            TestCase(
                id="ticket_summary_1",
                category="Ticket Summarization",
                name="VPN Connection Issue",
                prompt="Summarize this ServiceNow ticket:\n\nINC0012345\nPriority: P2\nReporter: John Smith\nDescription: User reports VPN connection drops every 10 minutes when working from home. Using Windows 11 and Cisco AnyConnect 4.10. Issue started after the recent Windows update KB5034441. User is a remote worker and needs stable VPN for all work tasks including accessing internal tools and databases.\nAdditional notes: User has tried restarting router and reinstalling VPN client.",
                expected_qualities=["concise", "captures key issues", "mentions technical details"],
                reference_output="P2 VPN connectivity issue for remote worker. Cisco AnyConnect disconnects every 10 mins on Win11 after KB5034441 update. User tried router restart and VPN reinstall without success.",
                difficulty="easy",
                tags=["ticket", "vpn", "network"]
            ),
            TestCase(
                id="ticket_summary_2",
                category="Ticket Summarization",
                name="Complex Multi-Issue Ticket",
                prompt="Summarize this ticket:\n\nINC0045678\nPriority: P1\nDescription: Multiple users in the Finance department (approx 15) cannot access SAP. Error: 'Connection timeout'. Also reporting slow Outlook performance and Teams calls dropping. Issue started at 9:15 AM. Network team confirms no planned changes. Users are on VLAN 120 in Building A, Floor 3. Some users can ping the SAP server but cannot establish connection.",
                expected_qualities=["identifies scope", "captures multiple issues", "notes investigation details"],
                difficulty="medium",
                tags=["ticket", "sap", "network", "multi-issue"]
            ),
            
            # Knowledge Article Generation
            TestCase(
                id="kb_article_1",
                category="Knowledge Article",
                name="Password Reset Guide",
                prompt="Write a knowledge article for: How to reset your Active Directory password using the self-service portal.",
                expected_qualities=["clear steps", "user-friendly", "complete process"],
                difficulty="easy",
                tags=["kb", "password", "self-service"]
            ),
            TestCase(
                id="kb_article_2",
                category="Knowledge Article",
                name="MFA Setup Guide",
                prompt="Create a knowledge article explaining: How to set up Microsoft Authenticator for Multi-Factor Authentication (MFA) when you get a new phone.",
                expected_qualities=["step-by-step", "includes prerequisites", "covers common issues"],
                difficulty="medium",
                tags=["kb", "mfa", "security"]
            ),
            
            # Troubleshooting
            TestCase(
                id="troubleshoot_1",
                category="Troubleshooting",
                name="Outlook Password Prompts",
                prompt="What are the troubleshooting steps for: Outlook keeps asking for password repeatedly after entering correct credentials?",
                expected_qualities=["systematic approach", "common causes", "actionable steps"],
                difficulty="medium",
                tags=["troubleshooting", "outlook", "authentication"]
            ),
            TestCase(
                id="troubleshoot_2",
                category="Troubleshooting",
                name="Slow Computer Diagnosis",
                prompt="Provide troubleshooting steps for: User reports their computer is extremely slow, taking 10+ minutes to boot and applications freeze frequently.",
                expected_qualities=["comprehensive", "prioritized steps", "considers multiple causes"],
                difficulty="medium",
                tags=["troubleshooting", "performance"]
            ),
            
            # SOP Generation
            TestCase(
                id="sop_1",
                category="SOP Generation",
                name="New Employee IT Onboarding",
                prompt="Create an SOP for: IT onboarding process for a new employee including equipment setup, account creation, and software installation.",
                expected_qualities=["ordered steps", "covers all aspects", "clear ownership"],
                difficulty="hard",
                tags=["sop", "onboarding", "process"]
            ),
            
            # Ticket Triage
            TestCase(
                id="triage_1",
                category="Ticket Triage",
                name="Priority Assignment",
                prompt="Assign priority and category for this ticket:\n\nSubject: CEO cannot access email\nDescription: CEO reports unable to open Outlook since this morning. Getting error 'Cannot connect to server'. CEO has board meeting in 2 hours and needs access to presentation materials in email.",
                expected_qualities=["correct priority", "appropriate category", "justification"],
                difficulty="easy",
                tags=["triage", "priority", "executive"]
            ),
            TestCase(
                id="triage_2",
                category="Ticket Triage",
                name="Complex Triage",
                prompt="Triage this ticket - assign priority, category, and suggest assignment group:\n\nSubject: Production database slow\nDescription: Application team reports production Oracle database queries taking 10x longer than normal. Affecting customer-facing application. Started 30 minutes ago. No recent deployments or changes.",
                expected_qualities=["high priority recognition", "correct categorization", "escalation awareness"],
                difficulty="medium",
                tags=["triage", "database", "production"]
            ),
            
            # Communication
            TestCase(
                id="comm_1",
                category="Communication",
                name="Outage Notification",
                prompt="Write an outage notification email for: Email system will be unavailable for maintenance on Saturday from 2 AM to 6 AM EST. All email access including mobile will be affected.",
                expected_qualities=["clear impact", "timing details", "professional tone"],
                difficulty="easy",
                tags=["communication", "outage", "email"]
            ),
            TestCase(
                id="comm_2",
                category="Communication",
                name="Incident Update",
                prompt="Write an incident update for stakeholders:\n\nIncident: Major network outage affecting Building B\nStatus: Ongoing for 2 hours\nImpact: 200 users cannot access any network resources\nCause: Identified as failed core switch\nETA: Replacement switch being configured, estimated 1 hour to restore",
                expected_qualities=["status clarity", "impact summary", "timeline", "next steps"],
                difficulty="medium",
                tags=["communication", "incident", "stakeholder"]
            ),
            
            # Technical Analysis
            TestCase(
                id="analysis_1",
                category="Technical Analysis",
                name="Root Cause Analysis",
                prompt="Perform root cause analysis:\n\nIncident: Production web server crashed\nTimeline:\n- 2:00 PM: Deployment of new code version\n- 2:15 PM: Memory usage started increasing\n- 2:45 PM: Server reached 95% memory\n- 2:50 PM: Server became unresponsive\n- 3:00 PM: Server automatically restarted\nLogs show repeated 'OutOfMemoryError' exceptions",
                expected_qualities=["identifies root cause", "timeline correlation", "recommendations"],
                difficulty="hard",
                tags=["analysis", "rca", "server"]
            ),
        ]
        
        for test in builtin_tests:
            self.test_cases[test.id] = test
    
    def get_tests_by_category(self, category: str) -> List[TestCase]:
        """Get all tests in a category."""
        return [t for t in self.test_cases.values() if t.category == category]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all categories."""
        return sorted(set(t.category for t in self.test_cases.values()))
    
    def add_custom_test(self, test: TestCase):
        """Add a custom test case."""
        self.test_cases[test.id] = test
    
    def record_response(self, response: TestResponse):
        """Record a model response."""
        self.responses.append(response)
    
    def record_comparison(self, comparison: ComparisonResult):
        """Record a comparison result."""
        self.comparisons.append(comparison)
    
    def get_responses_for_test(self, test_id: str) -> List[TestResponse]:
        """Get all responses for a specific test."""
        return [r for r in self.responses if r.test_id == test_id]
    
    def get_tuning_effectiveness_metrics(self) -> Dict[str, Any]:
        """Calculate overall tuning effectiveness metrics."""
        if not self.comparisons:
            return {}
        
        tuned_wins = sum(1 for c in self.comparisons if c.winner == "tuned")
        base_wins = sum(1 for c in self.comparisons if c.winner == "base")
        ties = sum(1 for c in self.comparisons if c.winner == "tie")
        total = len(self.comparisons)
        
        # Average improvement score
        avg_improvement = sum(c.improvement_score for c in self.comparisons) / total
        
        # Category breakdown
        category_stats = {}
        for comp in self.comparisons:
            cat = comp.test_case.category
            if cat not in category_stats:
                category_stats[cat] = {"tuned_wins": 0, "base_wins": 0, "ties": 0, "improvements": []}
            
            if comp.winner == "tuned":
                category_stats[cat]["tuned_wins"] += 1
            elif comp.winner == "base":
                category_stats[cat]["base_wins"] += 1
            else:
                category_stats[cat]["ties"] += 1
            
            category_stats[cat]["improvements"].append(comp.improvement_score)
        
        # Calculate averages per category
        for cat, stats in category_stats.items():
            stats["avg_improvement"] = sum(stats["improvements"]) / len(stats["improvements"])
            del stats["improvements"]
        
        return {
            "total_comparisons": total,
            "tuned_wins": tuned_wins,
            "base_wins": base_wins,
            "ties": ties,
            "tuned_win_rate": tuned_wins / total if total > 0 else 0,
            "avg_improvement_score": avg_improvement,
            "category_breakdown": category_stats,
        }
    
    def get_rating_summary(self) -> Dict[str, Any]:
        """Get summary of all ratings."""
        base_ratings = [r for r in self.responses if r.model_type == "base" and r.average_rating()]
        tuned_ratings = [r for r in self.responses if r.model_type == "tuned" and r.average_rating()]
        
        def calc_stats(responses):
            if not responses:
                return None
            ratings = [r.average_rating() for r in responses if r.average_rating()]
            if not ratings:
                return None
            return {
                "count": len(ratings),
                "average": sum(ratings) / len(ratings),
                "min": min(ratings),
                "max": max(ratings),
            }
        
        return {
            "base_model": calc_stats(base_ratings),
            "tuned_model": calc_stats(tuned_ratings),
            "total_responses": len(self.responses),
            "rated_responses": len([r for r in self.responses if r.average_rating()]),
        }
    
    def save_results(self, filename: Optional[str] = None):
        """Save all results to file."""
        if filename is None:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.storage_path / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "responses": [r.to_dict() for r in self.responses],
            "comparisons": [
                {
                    "test_id": c.test_case.id,
                    "test_name": c.test_case.name,
                    "base_response": c.base_response.to_dict(),
                    "tuned_response": c.tuned_response.to_dict(),
                    "winner": c.winner,
                    "improvement_score": c.improvement_score,
                }
                for c in self.comparisons
            ],
            "effectiveness_metrics": self.get_tuning_effectiveness_metrics(),
            "rating_summary": self.get_rating_summary(),
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def load_results(self, filepath: str):
        """Load results from file."""
        with open(filepath) as f:
            data = json.load(f)
        
        # Reconstruct responses
        for r_data in data.get("responses", []):
            response = TestResponse(
                test_id=r_data["test_id"],
                model_name=r_data["model_name"],
                model_type=r_data["model_type"],
                adapter_name=r_data.get("adapter_name"),
                response=r_data["response"],
                latency_ms=r_data["latency_ms"],
                timestamp=r_data.get("timestamp", ""),
            )
            
            ratings = r_data.get("ratings", {})
            response.relevance_rating = ratings.get("relevance")
            response.accuracy_rating = ratings.get("accuracy")
            response.quality_rating = ratings.get("quality")
            response.helpfulness_rating = ratings.get("helpfulness")
            response.overall_rating = ratings.get("overall")
            response.notes = r_data.get("notes", "")
            
            self.responses.append(response)
        
        logger.info(f"Loaded {len(self.responses)} responses from {filepath}")


def calculate_improvement_score(base_ratings: Dict[str, int], tuned_ratings: Dict[str, int]) -> float:
    """
    Calculate improvement score between base and tuned model.
    Returns value between -1 (tuned worse) and 1 (tuned better).
    """
    base_values = [v for v in base_ratings.values() if v is not None]
    tuned_values = [v for v in tuned_ratings.values() if v is not None]
    
    if not base_values or not tuned_values:
        return 0.0
    
    base_avg = sum(base_values) / len(base_values)
    tuned_avg = sum(tuned_values) / len(tuned_values)
    
    # Normalize to -1 to 1 scale (ratings are 1-5)
    diff = tuned_avg - base_avg  # Range: -4 to 4
    return diff / 4.0  # Normalize to -1 to 1


def determine_winner(base_response: TestResponse, tuned_response: TestResponse) -> Tuple[str, float]:
    """Determine winner based on ratings."""
    base_avg = base_response.average_rating()
    tuned_avg = tuned_response.average_rating()
    
    if base_avg is None or tuned_avg is None:
        return "tie", 0.0
    
    improvement = (tuned_avg - base_avg) / 4.0  # Normalize
    
    # Need significant difference to declare winner
    if abs(tuned_avg - base_avg) < 0.5:
        return "tie", improvement
    elif tuned_avg > base_avg:
        return "tuned", improvement
    else:
        return "base", improvement
