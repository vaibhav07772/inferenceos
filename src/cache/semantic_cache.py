"""InferenceOS — Semantic Cache (Redis + ChromaDB)"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# Silence chromadb telemetry
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rich.console import Console

console = Console()

# ─────────────────────────────────────
# Config
# ─────────────────────────────────────
CACHE_DIR = "./data/cache/chroma"
COLLECTION_NAME = "inference_cache"
DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_TTL_SECONDS = 3600


@dataclass
class CacheEntry:
    """Single cache entry"""
    cache_id: str
    query: str
    response: str
    tier: str
    model: str
    cost_usd: float
    latency_seconds: float
    created_at: str
    hit_count: int = 0


@dataclass
class CacheResult:
    """Result of cache lookup"""
    hit: bool
    entry: Optional[CacheEntry] = None
    similarity: float = 0.0
    latency_ms: float = 0.0
    savings_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hit": self.hit,
            "similarity": round(self.similarity, 3),
            "latency_ms": round(self.latency_ms, 2),
            "savings_usd": round(self.savings_usd, 6),
            "entry": {
                "query": self.entry.query,
                "tier": self.entry.tier,
                "model": self.entry.model,
                "hit_count": self.entry.hit_count,
            } if self.entry else None,
        }


# ─────────────────────────────────────
# Semantic Cache
# ─────────────────────────────────────
class SemanticCache:
    """
    Semantic cache using ChromaDB for similarity search.
    Redis alternative — 100% local, no external dependencies.
    """

    def __init__(
        self,
        cache_dir: str = CACHE_DIR,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        console.print("[cyan]🔧 Initializing Semantic Cache...[/cyan]")

        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.total_savings_usd = 0.0

        # ChromaDB setup
        os.makedirs(cache_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=cache_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embed_fn,
            )
            console.print(
                f"[green]✅ Semantic Cache ready "
                f"({self.collection.count()} entries)[/green]"
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            console.print("[green]✅ Semantic Cache ready (empty)[/green]")

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _is_expired(self, created_at: str) -> bool:
        """Check if cache entry is expired"""
        try:
            created = datetime.fromisoformat(created_at)
            age = (datetime.now() - created).total_seconds()
            return age > self.ttl_seconds
        except Exception:
            return False

    def get(self, query: str) -> CacheResult:
        """Look up similar query in cache"""
        start = time.time()

        if self.collection.count() == 0:
            self.misses += 1
            return CacheResult(hit=False, latency_ms=(time.time() - start) * 1000)

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=1,
            )

            if not results["documents"][0]:
                self.misses += 1
                return CacheResult(hit=False, latency_ms=(time.time() - start) * 1000)

            # Distance → similarity (cosine: distance in [0, 2])
            distance = results["distances"][0][0]
            similarity = 1.0 - (distance / 2.0)

            if similarity < self.similarity_threshold:
                self.misses += 1
                return CacheResult(
                    hit=False,
                    similarity=similarity,
                    latency_ms=(time.time() - start) * 1000,
                )

            # Parse metadata
            meta = results["metadatas"][0][0]

            # Check TTL
            if self._is_expired(meta.get("created_at", "")):
                console.print("[dim]⚠ Cache entry expired[/dim]")
                self.misses += 1
                return CacheResult(
                    hit=False,
                    similarity=similarity,
                    latency_ms=(time.time() - start) * 1000,
                )

            # Cache hit!
            entry = CacheEntry(
                cache_id=results["ids"][0][0],
                query=meta.get("original_query", ""),
                response=meta.get("response", ""),
                tier=meta.get("tier", ""),
                model=meta.get("model", ""),
                cost_usd=meta.get("cost_usd", 0.0),
                latency_seconds=meta.get("latency_seconds", 0.0),
                created_at=meta.get("created_at", ""),
                hit_count=meta.get("hit_count", 0) + 1,
            )

            # Update hit count
            self.collection.update(
                ids=[entry.cache_id],
                metadatas=[{**meta, "hit_count": entry.hit_count}],
            )

            self.hits += 1
            self.total_savings_usd += entry.cost_usd

            return CacheResult(
                hit=True,
                entry=entry,
                similarity=similarity,
                latency_ms=(time.time() - start) * 1000,
                savings_usd=entry.cost_usd,
            )

        except Exception as e:
            console.print(f"[yellow]⚠ Cache lookup failed: {e}[/yellow]")
            self.misses += 1
            return CacheResult(hit=False, latency_ms=(time.time() - start) * 1000)

    def set(
        self,
        query: str,
        response: str,
        tier: str,
        model: str,
        cost_usd: float,
        latency_seconds: float,
    ) -> str:
        """Store query-response pair in cache"""
        cache_id = self._hash(query)

        try:
            # Check if exists → update
            existing = self.collection.get(ids=[cache_id])

            metadata = {
                "original_query": query[:500],
                "response": response[:2000],
                "tier": tier,
                "model": model,
                "cost_usd": cost_usd,
                "latency_seconds": latency_seconds,
                "created_at": datetime.now().isoformat(),
                "hit_count": 0,
            }

            if existing["ids"]:
                # Update
                self.collection.update(
                    ids=[cache_id],
                    documents=[query],
                    metadatas=[metadata],
                )
            else:
                # Insert
                self.collection.add(
                    ids=[cache_id],
                    documents=[query],
                    metadatas=[metadata],
                )

            return cache_id

        except Exception as e:
            console.print(f"[yellow]⚠ Cache write failed: {e}[/yellow]")
            return ""

    def clear(self):
        """Clear all cache entries"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self.hits = 0
            self.misses = 0
            self.total_savings_usd = 0.0
            console.print("[green]✅ Cache cleared[/green]")
        except Exception as e:
            console.print(f"[red]❌ Cache clear failed: {e}[/red]")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0

        return {
            "total_entries": self.collection.count(),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
            "total_savings_usd": round(self.total_savings_usd, 6),
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds,
        }


# ─────────────────────────────────────
# Test
# ─────────────────────────────────────
if __name__ == "__main__":
    console.print("\n" + "="*70)
    console.print("[bold magenta]InferenceOS — Semantic Cache Test[/bold magenta]")
    console.print("="*70)

    cache = SemanticCache(similarity_threshold=0.85)

    # Clear old entries
    cache.clear()

    # ─── Test 1: Store entries ───
    console.print("\n[bold cyan]1. Storing cache entries...[/bold cyan]")

    cache.set(
        query="What is 2+2?",
        response="4",
        tier="small",
        model="openai/gpt-oss-20b",
        cost_usd=0.000020,
        latency_seconds=0.53,
    )

    cache.set(
        query="What is the capital of France?",
        response="Paris",
        tier="small",
        model="openai/gpt-oss-20b",
        cost_usd=0.000018,
        latency_seconds=0.48,
    )

    console.print(f"  Stored: {cache.collection.count()} entries")

    # ─── Test 2: Exact match ───
    console.print("\n[bold cyan]2. Testing exact match...[/bold cyan]")

    result = cache.get("What is 2+2?")
    if result.hit:
        console.print(f"  [green]✅ HIT[/green] (similarity: {result.similarity:.3f})")
        console.print(f"  Response: {result.entry.response}")
        console.print(f"  [dim]Latency: {result.latency_ms:.1f}ms (saved ${result.savings_usd:.6f})[/dim]")
    else:
        console.print(f"  [red]❌ MISS[/red] (similarity: {result.similarity:.3f})")

    # ─── Test 3: Semantic match ───
    console.print("\n[bold cyan]3. Testing semantic match...[/bold cyan]")

    result = cache.get("What does 2 + 2 equal?")  # Different wording
    if result.hit:
        console.print(f"  [green]✅ SEMANTIC HIT[/green] (similarity: {result.similarity:.3f})")
        console.print(f"  Matched: '{result.entry.query}'")
        console.print(f"  Response: {result.entry.response}")
    else:
        console.print(f"  [yellow]⚠ MISS[/yellow] (similarity: {result.similarity:.3f})")
        console.print(f"  [dim]Similarity too low — would miss this query[/dim]")

    # ─── Test 4: Miss (unrelated query) ───
    console.print("\n[bold cyan]4. Testing miss (unrelated query)...[/bold cyan]")

    result = cache.get("Explain quantum mechanics")
    if result.hit:
        console.print(f"  [red]❌ FALSE HIT[/red] (similarity: {result.similarity:.3f})")
    else:
        console.print(f"  [green]✅ MISS (expected)[/green] (similarity: {result.similarity:.3f})")

    # ─── Stats ───
    console.print("\n[bold cyan]📊 Cache Statistics:[/bold cyan]")
    stats = cache.stats()
    for k, v in stats.items():
        console.print(f"  {k}: {v}")

    console.print(f"\n[bold green]✅ Semantic Cache ready[/bold green]")