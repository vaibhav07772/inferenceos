"""InferenceOS — Multi-Tier LLM Client"""
import os
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type,
)
from rich.console import Console

load_dotenv()
console = Console()

# ─────────────────────────────────────
# Model Tiers (Groq current models — 2026)
# ─────────────────────────────────────
MODEL_TIERS = {
    "small": {
        "name": os.getenv("SMALL_MODEL", "openai/gpt-oss-20b"),
        "cost_per_1m_input": 0.10,
        "cost_per_1m_output": 0.30,
        "context_window": 32768,
    },
    "medium": {
        "name": os.getenv("MEDIUM_MODEL", "qwen/qwen3.6-27b"),
        "cost_per_1m_input": 0.20,
        "cost_per_1m_output": 0.50,
        "context_window": 131072,
    },
    "large": {
        "name": os.getenv("LARGE_MODEL", "openai/gpt-oss-120b"),
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
        "context_window": 131072,
    },
}


class LLMClient:
    """Multi-tier LLM client with cost tracking"""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing in .env")
        self.client = Groq(api_key=api_key)
        self._verify_models()

    def _verify_models(self):
        """Verify configured models are available"""
        try:
            available = {m.id for m in self.client.models.list().data}
            for tier_name, config in MODEL_TIERS.items():
                if config["name"] not in available:
                    console.print(
                        f"[yellow]⚠ Tier '{tier_name}' model "
                        f"'{config['name']}' not available[/yellow]"
                    )
        except Exception as e:
            console.print(f"[yellow]⚠ Could not verify models: {e}[/yellow]")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(
        self,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Raw API call with retry"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response

    def infer(
        self,
        prompt: str,
        tier: str = "small",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """Run inference on a specific tier"""

        if tier not in MODEL_TIERS:
            raise ValueError(f"Unknown tier: {tier}. Use: small/medium/large")

        tier_config = MODEL_TIERS[tier]
        model_name = tier_config["name"]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()

        try:
            response = self._call_api(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.time() - start

            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

            cost = (
                (input_tokens / 1_000_000) * tier_config["cost_per_1m_input"] +
                (output_tokens / 1_000_000) * tier_config["cost_per_1m_output"]
            )

            return {
                "success": True,
                "tier": tier,
                "model": model_name,
                "response": response.choices[0].message.content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": cost,
                "latency_seconds": latency,
            }

        except Exception as e:
            latency = time.time() - start
            return {
                "success": False,
                "tier": tier,
                "model": model_name,
                "error": str(e)[:200],
                "latency_seconds": latency,
            }


# ─────────────────────────────────────
# Test
# ─────────────────────────────────────
if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]InferenceOS — LLM Client Test[/bold magenta]")
    console.print("="*60)

    client = LLMClient()

    test_cases = [
        ("small", "What is 2+2? Answer in one word."),
        ("medium", "Explain what an API is in one sentence."),
        ("large", "Write a haiku about cloud computing."),
    ]

    for tier, prompt in test_cases:
        console.print(f"\n[cyan]Tier:[/cyan] {tier}")
        console.print(f"[dim]Prompt: {prompt[:60]}[/dim]")

        result = client.infer(prompt, tier=tier)

        if result["success"]:
            console.print(f"[green]✅ Response:[/green] {result['response'][:100]}")
            console.print(
                f"[dim]Tokens: {result['total_tokens']} | "
                f"Cost: ${result['cost_usd']:.6f} | "
                f"Latency: {result['latency_seconds']:.2f}s[/dim]"
            )
        else:
            console.print(f"[red]❌ Failed: {result['error']}[/red]")

        console.print("-" * 60)

    console.print(f"\n[bold green]✅ LLM Client ready[/bold green]")