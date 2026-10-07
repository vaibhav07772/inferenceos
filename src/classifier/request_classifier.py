"""InferenceOS — Request Classifier (complexity scoring)"""
import re
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console

console = Console()


# ─────────────────────────────────────
# Enums
# ─────────────────────────────────────
class Complexity(str, Enum):
    SIMPLE = "SIMPLE"
    MODERATE = "MODERATE"
    COMPLEX = "COMPLEX"


@dataclass
class ClassificationResult:
    """Result of request classification"""
    complexity: Complexity
    score: float
    recommended_tier: str
    reasons: list = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "score": round(self.score, 3),
            "recommended_tier": self.recommended_tier,
            "reasons": self.reasons,
            "features": self.features,
            "latency_ms": round(self.latency_ms, 2),
        }


# ─────────────────────────────────────
# Heuristic Patterns
# ─────────────────────────────────────
SIMPLE_PATTERNS = [
    r"^(what|who|when|where)\s+is\s+",
    r"^(what|who|when|where)'s\s+",
    r"^define\s+",
    r"^translate\s+",
    r"^\d+\s*[\+\-\*\/]\s*\d+",
    r"^(yes|no)\s*\??$",
]

COMPLEX_PATTERNS = [
    r"\b(analyze|evaluate|compare|critique|assess)\b",
    r"\b(design|architect|plan|strategy)\b",
    r"\b(implement|code|build|create)\b.*\b(system|app|framework)\b",
    r"\b(explain|describe)\b.*\b(in detail|thoroughly|comprehensively)\b",
    r"\b(step.by.step|comprehensive|detailed)\b",
    r"\b(reasoning|logic|proof|derive)\b",
    r"\bwhy\b.*\b(and|but|however)\b",
]

# Extended complexity indicators (v2)
COMPLEX_KEYWORDS = {
    # Analysis & reasoning
    "analyze", "evaluate", "compare", "contrast", "assess",
    "critique", "review", "reason", "derive", "prove",

    # Design & build
    "design", "architect", "implement", "build", "create",
    "refactor", "optimize", "debug", "develop",

    # Strategy & planning
    "strategy", "plan", "roadmap", "approach", "methodology",

    # Depth indicators
    "comprehensive", "detailed", "in detail", "in depth",
    "thoroughly", "step-by-step", "elaborate",

    # Technical
    "framework", "system", "architecture", "algorithm",
    "trade-off", "trade-offs", "tradeoff",
}

SIMPLE_KEYWORDS = {
    "what is", "who is", "when was", "where is",
    "define", "translate", "spell", "capital of",
    "how many", "calculate",
}


# ─────────────────────────────────────
# Classifier
# ─────────────────────────────────────
class RequestClassifier:
    """Classifies request complexity for routing"""

    def __init__(self):
        self.simple_compiled = [re.compile(p, re.IGNORECASE) for p in SIMPLE_PATTERNS]
        self.complex_compiled = [re.compile(p, re.IGNORECASE) for p in COMPLEX_PATTERNS]

    def _extract_features(self, text: str) -> Dict[str, Any]:
        """Extract complexity features from text"""
        text_lower = text.lower()
        words = text.split()

        length = len(text)
        word_count = len(words)
        sentence_count = max(1, text.count(".") + text.count("?") + text.count("!"))

        # Keyword presence
        complex_kw = [kw for kw in COMPLEX_KEYWORDS if kw in text_lower]
        simple_kw = [kw for kw in SIMPLE_KEYWORDS if kw in text_lower]

        # Regex pattern matches
        simple_pattern_matches = sum(1 for p in self.simple_compiled if p.search(text))
        complex_pattern_matches = sum(1 for p in self.complex_compiled if p.search(text))

        return {
            "length": length,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": length / max(1, word_count),
            "complex_keywords": complex_kw,
            "simple_keywords": simple_kw,
            "complex_pattern_matches": complex_pattern_matches,
            "simple_pattern_matches": simple_pattern_matches,
            "has_question": "?" in text,
            "has_multiple_questions": text.count("?") > 1,
            "has_code": any(c in text for c in ["```", "def ", "class ", "function"]),
        }

    def _compute_score(self, features: Dict[str, Any]) -> float:
        """
        Compute complexity score 0.0 (simple) → 1.0 (complex).
        Rebalanced weights (v2).
        """
        score = 0.0

        # ─── Length-based (0-0.30) ───
        wc = features["word_count"]
        if wc < 8:
            length_score = 0.0
        elif wc < 20:
            length_score = 0.05
        elif wc < 50:
            length_score = 0.15
        elif wc < 100:
            length_score = 0.25
        else:
            length_score = 0.30
        score += length_score

        # ─── Complex keywords (0-0.50) ───
        complex_kw_count = len(features["complex_keywords"])
        score += min(complex_kw_count * 0.20, 0.50)

        # ─── Complex regex patterns (0-0.30) ───
        score += min(features["complex_pattern_matches"] * 0.15, 0.30)

        # ─── Simple indicators (subtract) ───
        score -= features["simple_pattern_matches"] * 0.20
        score -= len(features["simple_keywords"]) * 0.10

        # ─── Code presence ───
        if features["has_code"]:
            score += 0.20

        # ─── Multiple questions ───
        if features["has_multiple_questions"]:
            score += 0.15

        # ─── Clamp ───
        return max(0.0, min(1.0, score))

    def _classify_from_score(self, score: float) -> Complexity:
        """Map score to complexity level"""
        if score < 0.30:
            return Complexity.SIMPLE
        elif score < 0.65:
            return Complexity.MODERATE
        else:
            return Complexity.COMPLEX

    def _recommend_tier(self, complexity: Complexity) -> str:
        """Map complexity to model tier"""
        return {
            Complexity.SIMPLE: "small",
            Complexity.MODERATE: "medium",
            Complexity.COMPLEX: "large",
        }[complexity]

    def classify(self, text: str) -> ClassificationResult:
        """Full classification"""
        start = time.time()

        if not text or not text.strip():
            return ClassificationResult(
                complexity=Complexity.SIMPLE,
                score=0.0,
                recommended_tier="small",
                reasons=["Empty input"],
                features={},
                latency_ms=0.0,
            )

        features = self._extract_features(text)
        score = self._compute_score(features)
        complexity = self._classify_from_score(score)
        tier = self._recommend_tier(complexity)

        # Build reasons
        reasons = []
        if features["complex_keywords"]:
            reasons.append(f"Complex keywords: {features['complex_keywords']}")
        if features["complex_pattern_matches"] > 0:
            reasons.append(f"Complex patterns: {features['complex_pattern_matches']}")
        if features["simple_pattern_matches"] > 0:
            reasons.append(f"Simple patterns: {features['simple_pattern_matches']}")
        if features["word_count"] > 100:
            reasons.append(f"Long input: {features['word_count']} words")
        if features["has_code"]:
            reasons.append("Contains code")
        if features["has_multiple_questions"]:
            reasons.append("Multiple questions")

        latency = (time.time() - start) * 1000

        return ClassificationResult(
            complexity=complexity,
            score=score,
            recommended_tier=tier,
            reasons=reasons or ["Default classification"],
            features=features,
            latency_ms=latency,
        )


# ─────────────────────────────────────
# Test
# ─────────────────────────────────────
if __name__ == "__main__":
    console.print("\n" + "="*70)
    console.print("[bold magenta]InferenceOS — Request Classifier Test[/bold magenta]")
    console.print("="*70)

    classifier = RequestClassifier()

    test_requests = [
        "What is 2+2?",
        "What is the capital of France?",
        "Define photosynthesis",
        "Explain what an API is in one sentence",
        "Analyze the trade-offs between microservices and monoliths in detail",
        "Design a distributed system for handling 1M requests per second",
        "Compare Python vs Rust for systems programming",
        "Write a step-by-step guide to implement JWT authentication",
        "Why does my code throw NullPointerException when I access .length()?",
        "What is the meaning of life, the universe, and everything? Also explain quantum mechanics",
    ]

    for req in test_requests:
        result = classifier.classify(req)

        color_map = {
            Complexity.SIMPLE: "green",
            Complexity.MODERATE: "yellow",
            Complexity.COMPLEX: "red",
        }
        color = color_map[result.complexity]

        console.print(f"\n[cyan]Request:[/cyan] {req[:70]}")
        console.print(
            f"  [{color}]Complexity: {result.complexity.value}[/{color}] "
            f"(score: {result.score:.3f})"
        )
        console.print(f"  [bold]→ Tier: {result.recommended_tier}[/bold]")
        if result.reasons:
            console.print(f"  [dim]Reasons: {result.reasons[0][:80]}[/dim]")

    console.print(f"\n[bold green]✅ Request Classifier ready[/bold green]")