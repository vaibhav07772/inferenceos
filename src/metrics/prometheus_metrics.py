"""InferenceOS — Prometheus Metrics"""
import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from rich.console import Console

console = Console()


# ─────────────────────────────────────
# Metrics Definitions
# ─────────────────────────────────────
class InferenceMetrics:
    """Central metrics registry for InferenceOS"""

    def __init__(self):
        # ─── Request counters ───
        self.requests_total = Counter(
            "inferenceos_requests_total",
            "Total inference requests",
            ["tier", "complexity", "cache_hit"],
        )

        self.errors_total = Counter(
            "inferenceos_errors_total",
            "Total inference errors",
            ["tier", "error_type"],
        )

        # ─── Latency histogram ───
        self.latency_seconds = Histogram(
            "inferenceos_latency_seconds",
            "End-to-end latency in seconds",
            ["tier", "cache_hit"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        # ─── Cost tracking ───
        self.cost_usd_total = Counter(
            "inferenceos_cost_usd_total",
            "Total cost in USD",
            ["tier", "model"],
        )

        self.cost_savings_usd_total = Counter(
            "inferenceos_cost_savings_usd_total",
            "Total savings from cache",
            ["tier"],
        )

        # ─── Token counters ───
        self.tokens_total = Counter(
            "inferenceos_tokens_total",
            "Total tokens processed",
            ["tier", "type"],  # type = input | output
        )

        # ─── Cache stats ───
        self.cache_hits_total = Counter(
            "inferenceos_cache_hits_total",
            "Total cache hits",
        )

        self.cache_misses_total = Counter(
            "inferenceos_cache_misses_total",
            "Total cache misses",
        )

        self.cache_size = Gauge(
            "inferenceos_cache_size",
            "Current cache entries",
        )

        self.cache_hit_rate = Gauge(
            "inferenceos_cache_hit_rate_pct",
            "Cache hit rate percentage",
        )

        # ─── System info ───
        self.active_requests = Gauge(
            "inferenceos_active_requests",
            "Currently active requests",
        )

        console.print("[green]✅ Prometheus metrics initialized[/green]")

    def record_request(
        self,
        tier: str,
        complexity: str,
        cache_hit: bool,
        latency_seconds: float,
        cost_usd: float,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_savings_usd: float = 0.0,
    ):
        """Record a single request"""
        cache_label = "true" if cache_hit else "false"

        # Request count
        self.requests_total.labels(
            tier=tier,
            complexity=complexity,
            cache_hit=cache_label,
        ).inc()

        # Latency
        self.latency_seconds.labels(
            tier=tier,
            cache_hit=cache_label,
        ).observe(latency_seconds)

        # Cache
        if cache_hit:
            self.cache_hits_total.inc()
            self.cost_savings_usd_total.labels(tier=tier).inc(cache_savings_usd)
        else:
            self.cache_misses_total.inc()

            # Only record cost + tokens on cache miss
            if cost_usd > 0:
                self.cost_usd_total.labels(tier=tier, model=model).inc(cost_usd)

            if input_tokens > 0:
                self.tokens_total.labels(tier=tier, type="input").inc(input_tokens)
            if output_tokens > 0:
                self.tokens_total.labels(tier=tier, type="output").inc(output_tokens)

    def record_error(self, tier: str, error_type: str):
        """Record an error"""
        self.errors_total.labels(tier=tier, error_type=error_type).inc()

    def update_cache_stats(self, size: int, hit_rate_pct: float):
        """Update cache gauges"""
        self.cache_size.set(size)
        self.cache_hit_rate.set(hit_rate_pct)

    def get_metrics_text(self) -> bytes:
        """Get Prometheus-format metrics"""
        return generate_latest()

    def get_content_type(self) -> str:
        return CONTENT_TYPE_LATEST


# ─────────────────────────────────────
# Global instance
# ─────────────────────────────────────
_metrics = None


def get_metrics() -> InferenceMetrics:
    """Get singleton metrics instance"""
    global _metrics
    if _metrics is None:
        _metrics = InferenceMetrics()
    return _metrics


# ─────────────────────────────────────
# Test
# ─────────────────────────────────────
if __name__ == "__main__":
    console.print("\n" + "="*70)
    console.print("[bold magenta]InferenceOS — Prometheus Metrics Test[/bold magenta]")
    console.print("="*70)

    metrics = get_metrics()

    console.print("\n[cyan]Simulating requests...[/cyan]")

    # Simulate requests
    metrics.record_request(
        tier="small",
        complexity="SIMPLE",
        cache_hit=False,
        latency_seconds=0.5,
        cost_usd=0.000020,
        model="openai/gpt-oss-20b",
        input_tokens=100,
        output_tokens=20,
    )

    metrics.record_request(
        tier="small",
        complexity="SIMPLE",
        cache_hit=True,
        latency_seconds=0.045,
        cost_usd=0.0,
        model="openai/gpt-oss-20b",
        cache_savings_usd=0.000020,
    )

    metrics.record_request(
        tier="medium",
        complexity="MODERATE",
        cache_hit=False,
        latency_seconds=1.2,
        cost_usd=0.000105,
        model="qwen/qwen3.8-27b",
        input_tokens=200,
        output_tokens=150,
    )

    metrics.record_error(tier="large", error_type="timeout")

    metrics.update_cache_stats(size=2, hit_rate_pct=50.0)

    # Print
    console.print("\n[cyan]Metrics output:[/cyan]\n")
    output = metrics.get_metrics_text().decode("utf-8")
    for line in output.split("\n"):
        if line and not line.startswith("#"):
            console.print(f"  {line}")

    console.print(f"\n[bold green]✅ Prometheus Metrics ready[/bold green]")