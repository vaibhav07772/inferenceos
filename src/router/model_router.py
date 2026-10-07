"""InferenceOS — Model Router (smart tier selection)"""
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console

from src.classifier.request_classifier import RequestClassifier, Complexity
from src.core.llm_client import LLMClient, MODEL_TIERS

console = Console()


# ─────────────────────────────────────
# Enums
# ─────────────────────────────────────
class RoutingStrategy(str, Enum):
    COST_OPTIMIZED = "COST_OPTIMIZED"
    BALANCED = "BALANCED"
    QUALITY_FIRST = "QUALITY_FIRST"
    FORCE_SMALL = "FORCE_SMALL"
    FORCE_MEDIUM = "FORCE_MEDIUM"
    FORCE_LARGE = "FORCE_LARGE"


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    tier: str
    model: str
    strategy: RoutingStrategy
    classifier_result: Dict[str, Any]
    routing_reason: str
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "model": self.model,
            "strategy": self.strategy.value,
            "classifier": self.classifier_result,
            "routing_reason": self.routing_reason,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class RoutedResponse:
    """Full routed response"""
    decision: RoutingDecision
    response: Dict[str, Any]
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "response": self.response,
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


# ─────────────────────────────────────
# Router
# ─────────────────────────────────────
class ModelRouter:
    """Intelligently routes requests to appropriate model tier"""

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.BALANCED):
        self.classifier = RequestClassifier()
        self.llm_client = LLMClient()
        self.default_strategy = strategy

    def _apply_strategy(
        self,
        recommended_tier: str,
        strategy: RoutingStrategy,
    ) -> tuple:
        """Apply routing strategy override"""
        if strategy == RoutingStrategy.FORCE_SMALL:
            return "small", "Strategy override: FORCE_SMALL"
        if strategy == RoutingStrategy.FORCE_MEDIUM:
            return "medium", "Strategy override: FORCE_MEDIUM"
        if strategy == RoutingStrategy.FORCE_LARGE:
            return "large", "Strategy override: FORCE_LARGE"

        if strategy == RoutingStrategy.QUALITY_FIRST:
            return "large", "Quality-first strategy — always largest tier"

        if strategy == RoutingStrategy.COST_OPTIMIZED:
            downgrade_map = {"large": "medium", "medium": "small", "small": "small"}
            downgraded = downgrade_map[recommended_tier]
            if downgraded != recommended_tier:
                return downgraded, f"Cost-optimized — downgraded from {recommended_tier}"
            return recommended_tier, "Already at lowest tier"

        # BALANCED — use classifier recommendation
        return recommended_tier, f"Classifier recommendation ({recommended_tier})"

    def route(
        self,
        prompt: str,
        strategy: Optional[RoutingStrategy] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> RoutedResponse:
        """Full pipeline: classify → route → infer"""
        start = time.time()

        strategy = strategy or self.default_strategy

        # ─── Step 1: Classify ───
        classification = self.classifier.classify(prompt)

        # ─── Step 2: Apply strategy ───
        tier, routing_reason = self._apply_strategy(
            classification.recommended_tier,
            strategy,
        )

        model = MODEL_TIERS[tier]["name"]

        decision = RoutingDecision(
            tier=tier,
            model=model,
            strategy=strategy,
            classifier_result=classification.to_dict(),
            routing_reason=routing_reason,
            latency_ms=(time.time() - start) * 1000,
        )

        # ─── Step 3: Infer ───
        response = self.llm_client.infer(
            prompt=prompt,
            tier=tier,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        total_latency = (time.time() - start) * 1000

        return RoutedResponse(
            decision=decision,
            response=response,
            total_latency_ms=total_latency,
        )


# ─────────────────────────────────────
# Test
# ─────────────────────────────────────
if __name__ == "__main__":
    console.print("\n" + "="*70)
    console.print("[bold magenta]InferenceOS — Model Router Test[/bold magenta]")
    console.print("="*70)

    router = ModelRouter(strategy=RoutingStrategy.BALANCED)

    test_cases = [
        ("What is 2+2?", "Simple math"),
        ("Explain what an API is in one sentence.", "Simple definition"),
        ("Design a distributed system for handling 1M requests per second.", "Complex design"),
    ]

    for prompt, label in test_cases:
        console.print(f"\n[bold cyan]Test: {label}[/bold cyan]")
        console.print(f"[dim]Prompt: {prompt[:70]}[/dim]")

        result = router.route(prompt, max_tokens=200)

        d = result.decision
        r = result.response

        tier_color = {"small": "green", "medium": "yellow", "large": "red"}[d.tier]

        console.print(
            f"  Classifier: {d.classifier_result['complexity']} "
            f"(score: {d.classifier_result['score']})"
        )
        console.print(f"  Routed: [{tier_color}]{d.tier}[/{tier_color}] → {d.model}")
        console.print(f"  Reason: {d.routing_reason}")

        if r["success"]:
            console.print(f"  [green]✅ Response:[/green] {r['response'][:80]}")
            console.print(
                f"  [dim]Tokens: {r['total_tokens']} | "
                f"Cost: ${r['cost_usd']:.6f} | "
                f"Latency: {r['latency_seconds']:.2f}s[/dim]"
            )
        else:
            console.print(f"  [red]❌ Failed: {r['error']}[/red]")

        console.print("-" * 70)

    console.print(f"\n[bold green]✅ Model Router ready[/bold green]")