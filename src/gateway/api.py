"""InferenceOS — FastAPI Gateway"""
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

# Add project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.router.model_router_v2 import ModelRouterV2
from src.router.model_router import RoutingStrategy

# ─────────────────────────────────────
# Global router (singleton)
# ─────────────────────────────────────
router: Optional[ModelRouterV2] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown"""
    global router
    print("🚀 Starting InferenceOS Gateway...")
    router = ModelRouterV2(cache_enabled=True, cache_threshold=0.92)
    print("✅ Gateway ready")
    yield
    print("👋 Shutting down Gateway...")


# ─────────────────────────────────────
# App
# ─────────────────────────────────────
app = FastAPI(
    title="InferenceOS API",
    description="Production LLM Inference & Optimization Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────
# Models
# ─────────────────────────────────────
class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="User prompt")
    strategy: Optional[str] = Field(default="BALANCED", description="Routing strategy")
    system_prompt: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=1, le=4000)
    use_cache: bool = Field(default=True)


class InferenceResponse(BaseModel):
    success: bool
    response: str
    tier: str
    model: str
    complexity: str
    complexity_score: float
    cache_hit: bool
    cache_similarity: float
    cache_savings_usd: float
    total_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class StatsResponse(BaseModel):
    cache: dict


# ─────────────────────────────────────
# Routes
# ─────────────────────────────────────
@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "InferenceOS API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "infer": "/v1/infer",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/v1/infer", response_model=InferenceResponse, tags=["Inference"])
async def infer(request: InferenceRequest):
    """
    Run inference with smart routing + semantic cache.

    Returns response with routing decision, cache info, and cost.
    """
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    # Parse strategy
    try:
        strategy = RoutingStrategy(request.strategy.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Use: {[s.value for s in RoutingStrategy]}",
        )

    try:
        result = router.route(
            prompt=request.prompt,
            strategy=strategy,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_cache=request.use_cache,
        )

        d = result.decision
        r = result.response

        return InferenceResponse(
            success=r["success"],
            response=r.get("response", r.get("error", "")),
            tier=d.tier,
            model=d.model,
            complexity=d.classifier_result["complexity"],
            complexity_score=d.classifier_result["score"],
            cache_hit=result.cache_hit,
            cache_similarity=result.cache_similarity,
            cache_savings_usd=result.cache_savings_usd,
            total_tokens=r.get("total_tokens", 0),
            cost_usd=r.get("cost_usd", 0.0),
            latency_ms=result.total_latency_ms,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/stats", response_model=StatsResponse, tags=["Info"])
async def stats():
    """Get cache + system statistics"""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    return StatsResponse(cache=router.cache_stats())


@app.delete("/v1/cache", tags=["Admin"])
async def clear_cache():
    """Clear semantic cache"""
    if router is None or not router.cache:
        raise HTTPException(status_code=503, detail="Cache not available")

    router.cache.clear()
    return {"status": "cache cleared"}


# ─────────────────────────────────────
# Run
# ─────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "src.gateway.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )