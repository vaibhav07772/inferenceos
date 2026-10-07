"""InferenceOS — Model Router v2 (with Semantic Cache)"""
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from rich.console import Console

from src.classifier.request_classifier import RequestClassifier
from src.core.llm_client import LLMClient, MODEL_TIERS
from src.cache.semantic_cache import SemanticCache
from src.router.model_router import (
    RoutingStrategy,
    RoutingDecision,
    RoutedResponse,
)

console = Console()


@dataclass
class CachedRoutedResponse:
    """Extended response with cache info"""
    decision: RoutingDecision
    response: Dict[str, Any]
    cache_hit: bool
    cache_similarity: float
    cache_savings_usd: float
    total_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "response": self.response,
            "cache": {
                "hit": self.cache_hit,
                "similarity": round(self.cache_similarity, 3),
                "savings_usd": round(self.cache_savings_usd, 6),
            },
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


class ModelRouterV2:
    """Router with semantic cache integration"""

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        cache_enabled: bool = True,
        cache_threshold: float = 0.92,
    ):
        console.print("[cyan]🔧 Initializing Router v2...[/cyan]")
        self.classifier = RequestClassifier()
        self.llm_client = LLMClient()
        self.default_strategy = strategy
        self.cache_enabled = cache_enabled

        if cache_enabled:
            self.cache = SemanticCache(similarity_threshold=cache_threshold)
        else:
            self.cache = None

        console.print("[green]✅ Router v2 ready[/green]")

    def _apply_strategy(
        self,
        recommended_tier: str,
        strategy: RoutingStrategy,
    ) -> tuple:
        """Apply routing strategy"""
        if strategy == RoutingStrategy.FORCE_SMALL:
            return "small", "Force SMALL"
        if strategy == RoutingStrategy.FORCE_MEDIUM:
            return "medium", "Force MEDIUM"
        if strategy == RoutingStrategy.FORCE_LARGE:
            return "large", "Force LARGE"
        if strategy == RoutingStrategy.QUALITY_FIRST:
            return "large", "Quality-first"
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            downgrade = {"large": "medium", "medium": "small", "small": "small"}
            return downgrade[recommended_tier], f"Cost-optimized → {downgrade[recommended_tier]}"
        return recommended_tier, f"Classifier → {recommended_tier}"

    def route(
        self,
        prompt: str,
        strategy: Optional[RoutingStrategy] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        use_cache: bool = True,
    ) -> CachedRoutedResponse:
        """Full pipeline: cache → classify → route → infer → cache store"""
        start = time.time()
        strategy = strategy or self.default_strategy

        # ─── Step 1: Cache Lookup ───
        if self.cache_enabled and use_cache:
            cache_result = self.cache.get(prompt)

            if cache_result.hit:
                # Return cached response
                classification = self.classifier.classify(prompt)
                cached_tier = cache_result.entry.tier
                cached_model = cache_result.entry.model

                decision = RoutingDecision(
                    tier=cached_tier,
                    model=cached_model,
                    strategy=strategy,
                    classifier_result=classification.to_dict(),
                    routing_reason=f"CACHE HIT (similarity: {cache_result.similarity:.3f})",
                    latency_ms=cache_result.latency_ms,
                )

                cached_response = {
                    "success": True,
                    "tier": cached_tier,
                    "model": cached_model,
                    "response": cache_result.entry.response,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_seconds": cache_result.latency_ms / 1000,
                    "cached": True,
                }

                return CachedRoutedResponse(
                    decision=decision,
                    response=cached_response,
                    cache_hit=True,
                    cache_similarity=cache_result.similarity,
                    cache_savings_usd=cache_result.savings_usd,
                    total_latency_ms=(time.time() - start) * 1000,
                )

        # ─── Step 2: Classify ───
        classification = self.classifier.classify(prompt)

        # ─── Step 3: Route ───
        tier, routing_reason = self._apply_strategy(
            classification.recommended_tier, strategy
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

        # ─── Step 4: Infer ───
        response = self.llm_client.infer(
            prompt=prompt,
            tier=tier,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # ─── Step 5: Store in Cache ───
        if self.cache_enabled and response.get("success"):
            self.cache.set(
                query=prompt,
                response=response["response"],
                tier=tier,
                model=model,
                cost_usd=response["cost_usd"],
                latency_seconds=response["latency_seconds"],
            )

        return CachedRoutedResponse(
            decision=decision,
            response=response,
            cache_hit=False,
            cache_similarity=0.0,
            cache_savings_usd=0.0,
            total_latency_ms=(time.time() - start) * 1000,
        )

    def cache_stats(self) -> Dict[str, Any]:
        if not self.cache_enabled:
            return {"cache_enabled": False}
        return self.cache.stats()


# ─────────────────────────────────────
# Test
# ─────────────────────────────────────
if __name__ == "__main__":
    console.print("\n" + "="*70)
    console.print("[bold magenta]InferenceOS — Router v2 (with Cache)[/bold magenta]")
    console.print("="*70)

    router = ModelRouterV2(cache_enabled=True, cache_threshold=0.85)
    router.cache.clear()

    test_scenarios = [
        ("What is 2+2?", "First request (cache miss)"),
        ("What is 2+2?", "Second request (exact cache hit)"),
        ("What does 2 + 2 equal?", "Third request (semantic cache hit)"),
        ("Design a distributed system.", "Different query (cache miss)"),
    ]

    for prompt, label in test_scenarios:
        console.print(f"\n[bold cyan]{label}[/bold cyan]")
        console.print(f"[dim]Prompt: {prompt}[/dim]")

        result = router.route(prompt, max_tokens=150)

        if result.cache_hit:
            console.print(f"  🎯 [green]CACHE HIT[/green] (similarity: {result.cache_similarity:.3f})")
            console.print(f"  [green]Saved:[/green] ${result.cache_savings_usd:.6f}")
        else:
            console.print(f"  🔄 [yellow]CACHE MISS[/yellow]")

        if result.response["success"]:
            console.print(f"  Response: {result.response['response'][:80]}")

        console.print(
            f"  [dim]Tier: {result.decision.tier} | "
            f"Tokens: {result.response['total_tokens']} | "
            f"Cost: ${result.response['cost_usd']:.6f} | "
            f"Total: {result.total_latency_ms:.0f}ms[/dim]"
        )
        console.print("-" * 70)

    # Stats
    console.print("\n[bold cyan]📊 Cache Statistics:[/bold cyan]")
    stats = router.cache_stats()
    for k, v in stats.items():
        console.print(f"  {k}: {v}")

    console.print(f"\n[bold green]✅ Router v2 ready[/bold green]")