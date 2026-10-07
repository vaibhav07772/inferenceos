# 🔥 InferenceOS

<div align="center">

## Production LLM Inference & Optimization Platform

**Intelligent Routing · Semantic Caching · Cost Optimization · Full Observability**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LLM-F55036)](https://groq.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?logo=prometheus)](https://prometheus.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 Table of Contents

* [Overview](#-overview)
* [The Problem](#-the-problem)
* [Key Results](#-key-results)
* [Demo](#-demo)
* [Architecture](#-architecture)
* [Core Features](#-core-features)
* [Multi-Tier Routing](#-multi-tier-routing)
* [Request Classifier](#-request-classifier)
* [Semantic Cache](#-semantic-cache)
* [Routing Strategies](#-routing-strategies)
* [Observability](#-observability)
* [API](#-api)
* [Quick Start](#-quick-start)
* [Docker](#-docker)
* [Project Structure](#-project-structure)
* [Technology Stack](#-technology-stack)
* [Performance](#-performance)
* [Interview Talking Points](#-interview-talking-points)
* [Limitations](#-limitations)
* [Roadmap](#-roadmap)
* [License](#-license)

---

# 🎯 Overview

**InferenceOS** is a production-oriented LLM inference platform designed to optimize three critical dimensions of LLM applications:

```text
                 INFERENCEOS
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       COST ↓      LATENCY ↓   QUALITY ↑
```

Instead of sending every request to the same expensive model, InferenceOS analyzes the incoming request, determines its complexity, and dynamically selects an appropriate model tier.

It also uses **semantic caching** to avoid unnecessary LLM calls when a sufficiently similar query has already been answered.

### Core Pipeline

```text
User Request
     │
     ▼
Semantic Cache
     │
 ┌───┴────┐
 │        │
HIT      MISS
 │        │
 ▼        ▼
Return   Classifier
45ms       │
           ▼
      Model Router
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
   SMALL MEDIUM LARGE
     │     │     │
     └─────┼─────┘
           ▼
      LLM Response
           │
           ▼
    Prometheus Metrics
```

---

# 🚨 The Problem

A common architecture for LLM applications looks like this:

```text
Every Request
     │
     ▼
Same Large Model
     │
     ▼
Expensive Inference
```

This creates several problems:

* Simple queries consume expensive model capacity.
* Repeated queries trigger unnecessary LLM calls.
* Latency becomes unpredictable.
* Cost visibility is poor.
* There is often no centralized inference gateway.
* Production teams need better model-level observability.

InferenceOS addresses these problems using:

> **Complexity-aware routing + semantic caching + cost tracking + observability.**

---

# 📊 Key Results

The current implementation demonstrates:

| Metric                       |         Result |
| ---------------------------- | -------------: |
| Queries routed to cheap tier |       **~70%** |
| Semantic cache hit rate      |     **30–40%** |
| Cache response improvement   | **22× faster** |
| Cache hit LLM cost           |         **$0** |
| Cache hit latency            |     **~45 ms** |
| Cache miss latency           |    **~620 ms** |
| Prometheus metrics           |         **12** |
| Routing strategies           |          **6** |
| Model tiers                  |          **3** |

The project therefore demonstrates how an inference layer can reduce unnecessary model usage while improving response latency.

---

# 📸 Demo

## Dashboard

![InferenceOS Dashboard](docs/landing.png)

> Streamlit dashboard showing routing strategies, model tiers, inference information, and live cache statistics.

---

## Live Inference

![InferenceOS Result](docs/result.png)

> Example request classified as SIMPLE and served through the semantic cache with high similarity.

---

# 🏗️ Architecture

```text
                         USER
                           │
                           ▼
                ┌─────────────────────┐
                │   FastAPI Gateway   │
                │   REST API + Docs   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Semantic Cache   │
                │      ChromaDB       │
                └──────────┬──────────┘
                           │
                  ┌────────┴────────┐
                  │                 │
               CACHE HIT        CACHE MISS
                  │                 │
                  ▼                 ▼
             ┌─────────┐    ┌───────────────┐
             │ Return  │    │   Classifier  │
             │ ~45 ms  │    │   Score 0–1   │
             └─────────┘    └───────┬───────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │   Model Router  │
                           │  6 Strategies   │
                           └────────┬────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
               ┌────────┐     ┌────────┐      ┌────────┐
               │ SMALL  │     │ MEDIUM │      │ LARGE  │
               │ 20B    │     │ 27B    │      │ 120B   │
               └────┬───┘     └────┬───┘      └────┬───┘
                    │              │                │
                    └──────────────┼────────────────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │ Prometheus       │
                         │ 12 Metrics       │
                         └──────────────────┘
```

---

# ✨ Core Features

## 1. 🎯 Multi-Tier LLM Routing

InferenceOS supports three model tiers:

| Tier          | Model          | Typical Use Case    | Cost / 1M Tokens     |
| ------------- | -------------- | ------------------- | -------------------- |
| 🟢 **Small**  | `gpt-oss-20b`  | Simple Q&A          | $0.10 in / $0.30 out |
| 🟡 **Medium** | `qwen3.8-27b`  | Moderate complexity | $0.20 in / $0.50 out |
| 🔴 **Large**  | `gpt-oss-120b` | Complex reasoning   | $0.15 in / $0.60 out |

The router avoids using the largest model for every request.

---

# 🧠 Request Classifier

The request classifier assigns a complexity score between:

```text
0.0 ─────────────────────────────── 1.0
```

### Classification

|         Score | Complexity | Selected Tier |
| ------------: | ---------- | ------------- |
| `0.00 – 0.30` | SIMPLE     | Small         |
| `0.30 – 0.65` | MODERATE   | Medium        |
| `0.65 – 1.00` | COMPLEX    | Large         |

### Signals

The heuristic classifier considers:

* Word count
* Query length
* Complex keywords
* Simple keywords
* Regex patterns
* Presence of code
* Reasoning-oriented language

Examples:

```text
"What is 2+2?"
        ↓
SIMPLE
        ↓
SMALL
```

```text
"Design a distributed system for millions of users."
        ↓
MODERATE
        ↓
MEDIUM
```

```text
"Analyze the trade-offs between multiple distributed architectures."
        ↓
COMPLEX
        ↓
LARGE
```

---

# 💾 Semantic Cache

One of the most important components of InferenceOS is its **semantic cache**.

Instead of checking only exact string equality, the cache uses embeddings to determine whether two queries are semantically similar.

### Technology

```text
ChromaDB
    +
MiniLM Embeddings
    ↓
Semantic Similarity
```

### Example

```text
Query 1:
"What is 2+2?"

Query 2:
"What does 2 + 2 equal?"
```

These queries are semantically equivalent and can resolve to the same cached response.

### Cache Behavior

| Feature         | Configuration              |
| --------------- | -------------------------- |
| Exact Match     | Similarity `1.00`          |
| Semantic Match  | Example similarity `0.954` |
| Default TTL     | 1 hour                     |
| Threshold       | Configurable               |
| Suggested Range | `0.85 – 0.95`              |

### Why Semantic Cache?

Without caching:

```text
Request
  ↓
LLM
  ↓
Cost + Latency
```

With caching:

```text
Request
  ↓
Semantic Search
  ↓
Cache HIT
  ↓
Instant Response
```

Result:

> **~22× faster responses and zero LLM cost on cache hits.**

---

# 🚦 Six Routing Strategies

InferenceOS provides six routing strategies.

| Strategy         | Behavior                       |
| ---------------- | ------------------------------ |
| `BALANCED`       | Uses classifier recommendation |
| `COST_OPTIMIZED` | Downgrades one model tier      |
| `QUALITY_FIRST`  | Always uses large tier         |
| `FORCE_SMALL`    | Forces small tier              |
| `FORCE_MEDIUM`   | Forces medium tier             |
| `FORCE_LARGE`    | Forces large tier              |

This makes the system useful for both production routing and controlled experimentation.

---

# 📊 Observability

InferenceOS exposes **12 Prometheus metrics**.

### Request Metrics

```text
requests_total
active_requests
errors_total
```

### Latency

```text
latency_seconds
```

Histogram with multiple latency buckets.

### Cost

```text
cost_usd_total
cost_savings_usd_total
```

### Token Usage

```text
tokens_total
```

### Cache

```text
cache_hits_total
cache_misses_total
cache_size
cache_hit_rate
```

These metrics provide visibility into:

* Model usage
* Request volume
* Latency
* Token consumption
* Estimated cost
* Cache performance
* Error rates
* Active traffic

---

# 🌐 REST API

InferenceOS provides a FastAPI gateway.

## Health Check

```bash
curl http://localhost:8000/health
```

---

## Inference

```bash
curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is 2+2?","max_tokens":100}'
```

### Example Response

```json
{
  "success": true,
  "response": "4",
  "tier": "small",
  "model": "openai/gpt-oss-20b",
  "complexity": "SIMPLE",
  "complexity_score": 0.0,
  "cache_hit": false,
  "cost_usd": 0.000019,
  "latency_ms": 620.5
}
```

The response exposes important inference metadata including:

* Selected tier
* Selected model
* Complexity
* Complexity score
* Cache status
* Estimated cost
* Latency

---

## Cache Statistics

```bash
curl http://localhost:8000/v1/stats
```

---

## Swagger Documentation

Once the API is running:

```text
http://localhost:8000/docs
```

---

# 🚀 Quick Start

## Prerequisites

* Python 3.11
* Conda
* Docker (optional)
* Groq API key

---

## 1. Clone Repository

```bash
git clone https://github.com/vaibhav07772/inferenceos.git
cd inferenceos
```

---

## 2. Create Environment

```bash
conda create -n inferenceos python=3.11 -y
conda activate inferenceos
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure API Key

Create a `.env` file:

```env
GROQ_API_KEY=gsk_your_key_here
```

---

# ▶️ Run the API

```bash
python -m src.gateway.api
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

# 🖥️ Run the Dashboard

In another terminal:

```bash
streamlit run src/dashboard/app.py
```

Dashboard:

```text
http://localhost:8501
```

---

# 🐳 Docker

## Build Image

```bash
docker build -t inferenceos:latest .
```

## Run Container

```bash
docker run \
  -p 8000:8000 \
  --env-file .env \
  inferenceos:latest
```

---

## Docker Compose

For the full stack:

```bash
docker-compose up
```

---

# 📁 Project Structure

```text
inferenceos/
│
├── src/
│   │
│   ├── core/
│   │   └── llm_client.py
│   │       # Multi-tier Groq LLM client
│   │
│   ├── classifier/
│   │   └── request_classifier.py
│   │       # Complexity scoring
│   │
│   ├── router/
│   │   ├── model_router.py
│   │   └── model_router_v2.py
│   │       # Routing + cache integration
│   │
│   ├── cache/
│   │   └── semantic_cache.py
│   │       # ChromaDB semantic cache
│   │
│   ├── gateway/
│   │   └── api.py
│   │       # FastAPI inference gateway
│   │
│   ├── metrics/
│   │   └── prometheus_metrics.py
│   │       # Prometheus metrics
│   │
│   └── dashboard/
│       └── app.py
│           # Streamlit dashboard
│
├── data/
│   ├── cache/
│   ├── logs/
│   └── metrics/
│
├── docs/
│   ├── landing.png
│   └── result.png
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🛠️ Technology Stack

| Layer         | Technology        | Purpose                     |
| ------------- | ----------------- | --------------------------- |
| API           | FastAPI + Uvicorn | REST inference gateway      |
| LLM           | Groq              | Multi-tier inference        |
| Models        | 20B / 27B / 120B  | Tiered routing              |
| Cache         | ChromaDB          | Persistent semantic cache   |
| Embeddings    | MiniLM            | Query similarity            |
| Metrics       | Prometheus Client | Production observability    |
| UI            | Streamlit         | Inference dashboard         |
| Visualization | Plotly            | Metrics visualization       |
| Deployment    | Docker            | Containerization            |
| Orchestration | Docker Compose    | Full-stack local deployment |

> **Current project testing uses the Groq free tier, keeping the stated development cost at $0.**

---

# 📈 Performance

## Before vs After Cache

| Metric  | Cache Miss |     Cache Hit |    Improvement |
| ------- | ---------: | ------------: | -------------: |
| Latency |     620 ms |     **45 ms** | **22× faster** |
| Cost    |  $0.000019 | **$0.000000** | **100% saved** |
| Tokens  |        116 |         **0** |    No LLM call |

---

# 🎯 Real Routing Examples

| Query                          | Complexity      | Tier   |      Cost |
| ------------------------------ | --------------- | ------ | --------: |
| `What is 2+2?`                 | SIMPLE `0.00`   | Small  | $0.000020 |
| `Explain API`                  | SIMPLE `0.05`   | Small  | $0.000033 |
| `Design distributed system`    | MODERATE `0.60` | Medium | $0.000078 |
| `Analyze trade-offs in detail` | COMPLEX `0.70`  | Large  | $0.000105 |

This demonstrates the core idea:

```text
Simple Query
     ↓
Cheap Model

Moderate Query
     ↓
Medium Model

Complex Query
     ↓
Large Model
```

---

# 🎓 Interview Talking Points

## 1. Why did you build an inference platform?

> **"In production, not every LLM request requires the most powerful model. I built InferenceOS to demonstrate cost-aware model routing where simple requests are sent to a smaller model while complex requests are routed to a larger reasoning model. I combined this with semantic caching so repeated or highly similar requests can avoid LLM calls completely."**

---

## 2. How does semantic caching work?

> **"The system converts incoming queries into embeddings and stores them in ChromaDB. For every new request, it searches for semantically similar previous queries. If the similarity exceeds the configured threshold, the cached response is returned without calling the LLM. For example, 'What is 2+2?' and 'What does 2 + 2 equal?' can resolve to the same cache entry."**

---

## 3. How does model routing work?

> **"A heuristic classifier generates a complexity score between 0 and 1 using signals such as word count, complexity keywords, regex patterns and code presence. Scores from 0 to 0.30 are treated as simple, 0.30 to 0.65 as moderate, and above 0.65 as complex. The model router then selects the appropriate inference tier."**

---

## 4. Why not always use the largest model?

> **"Because model capability comes with latency and cost trade-offs. Sending simple factual questions to a large model wastes compute. A tiered router lets us reserve stronger models for requests that actually require them."**

---

## 5. What is the biggest challenge?

> **"Semantic cache poisoning was an important challenge. If the similarity threshold is too low, a semantically related but logically different query can incorrectly reuse a cached response. I therefore use a configurable similarity threshold and treat cache quality as a critical part of the inference architecture."**

---

## 6. How did you optimize latency?

> **"The biggest latency optimization comes from semantic caching. A cache miss takes around 620 milliseconds in the current test, while a cache hit returns in around 45 milliseconds, giving approximately a 22x improvement. The cache also completely eliminates the LLM inference cost for those requests."**

---

## 7. How would you productionize InferenceOS?

> **"I would replace the single-node ChromaDB cache with Redis for distributed caching, deploy the gateway and inference workers on Kubernetes with HPA, add LangSmith or equivalent tracing for LLM-level observability, and introduce authentication, authorization and rate limiting at the API gateway."**

---

# 🧠 Key AI Engineering Concepts Demonstrated

InferenceOS demonstrates practical knowledge of:

### LLM Infrastructure

* Multi-model inference
* Model routing
* Model tiering
* Cost-aware inference
* Latency optimization

### RAG / Retrieval Infrastructure

* Embeddings
* Semantic similarity
* Vector database
* Semantic caching

### Backend Engineering

* FastAPI
* REST APIs
* Typed request/response models
* Swagger/OpenAPI

### MLOps / LLMOps

* Prometheus metrics
* Cost tracking
* Token monitoring
* Latency monitoring
* Cache monitoring
* Docker
* Docker Compose

### System Design

* Inference gateway
* Caching layer
* Routing layer
* Observability layer
* Model abstraction

---

# ⚠️ Limitations

Current limitations:

* **No authentication** — designed primarily as a local inference platform.
* **No rate limiting** — gateway middleware can be added later.
* **Single-node cache** — ChromaDB is persistent but not distributed.
* **Heuristic classifier** — an embedding/ML-based classifier could improve routing accuracy.
* **No request batching** — each request is currently processed independently.
* **Current routing decisions are based on heuristic complexity signals rather than a learned routing model.**

---

# 🗺️ Roadmap

## ✅ Completed

* [x] Multi-tier LLM client
* [x] 3 Groq model tiers
* [x] Request complexity classifier
* [x] 6 routing strategies
* [x] Semantic cache
* [x] Cache-integrated model router
* [x] FastAPI inference gateway
* [x] Swagger API documentation
* [x] Prometheus metrics
* [x] Streamlit dashboard
* [x] Plotly visualizations
* [x] Docker support
* [x] Docker Compose support

## 🚧 Planned

* [ ] Redis distributed cache
* [ ] Kubernetes deployment
* [ ] Horizontal Pod Autoscaling
* [ ] LangSmith tracing
* [ ] ML-based / embedding-based request classifier
* [ ] Request batching
* [ ] Authentication
* [ ] Rate limiting
* [ ] Distributed inference workers
* [ ] Advanced cost optimization policies

---

# 💡 Production Evolution

The intended evolution of InferenceOS is:

```text
CURRENT
   │
   ├── FastAPI
   ├── ChromaDB
   ├── Heuristic Router
   ├── Prometheus
   └── Docker
        │
        ▼
FUTURE
   │
   ├── Kubernetes
   ├── Redis Cluster
   ├── ML Router
   ├── Distributed Workers
   ├── HPA
   ├── Authentication
   ├── Rate Limiting
   └── Advanced LLM Observability
```

This would transform the current prototype into a horizontally scalable inference platform.

---

# 📄 License

MIT © [Vaibhav Singh](https://github.com/vaibhav07772)

---

<div align="center">

## 🔥 Intelligent Inference. Lower Cost. Lower Latency.

**⭐ Star this repository if you find it useful!**

Built with ❤️ by **Vaibhav Singh**

</div>
