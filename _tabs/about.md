---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

<link rel="icon" type="image/x-icon" href="{{ '/assets/img/favicons/ild01@4x.ico' | relative_url }}">

## About Me

ML Engineer specialized in MLOps and cloud‑native AI systems. I design, containerize, and operate ML services that meet production SLOs for latency, throughput, availability, and cost. I bridge data science with platform engineering: reproducible training, efficient serving (FastAPI/BentoML/vLLM/LiteLLM), robust observability, and safe CI/CD on Docker/Kubernetes/ECS.

## Operating Principles

- Reproducibility first: versioned data/model/code with MLflow and immutable artifacts
- Observability by design: metrics for drift, latency, errors; actionable alerts
- Reliability & SLOs: degrade gracefully, fallback policies, shadow/canary releases
- Performance & scale: adaptive batching, worker isolation, autoscaling on K8s/ECS
- Cost efficiency: throughput per dollar, right‑sizing, caching and cold‑start control
- Security & governance: access control, audit trails, policy‑driven deployments

## Architecture Highlights

- Model serving: FastAPI/BentoML services with separate ML workers and web workers for concurrency; autoscale via Kubernetes/ECS; adaptive batching for GPUs/CPUs
- Experiment tracking: MLflow runs, artifacts, and metrics to ensure comparable, reproducible iterations across datasets and hyperparameters
- Drift & performance monitoring: KS‑tests on match scores and response latency; thresholded alerts and rollback/traffic shifting when quality degrades
- Storage & data: S3/MinIO object storage with deterministic preprocessing; schema validation and data quality gates
- CI/CD: GitHub Actions building and testing containers; smoke tests, blue/green or canary strategies for safe rollouts
- Eventing: MQTT where event‑driven triggers decouple retraining and inference pipelines
- LLMOps: vLLM/LiteLLM for efficient token throughput; prompt/version management, caching, and cost/latency accounting

## Experience Snapshots

- DialFlow (Winner 2025): Gen‑AI voice agent using Twilio, FastAPI, Redis, ElevenLabs, and LangChain; streaming pipeline, prompt governance, and service KPIs for reliability and cost control
- UM6P — DICE | DATALAB: built forensic image feature‑matching service; template code for retrainable workflows; online drift monitoring on scores and latency with KS tests
- 3D Smart Factory: NLP pipeline with S3 storage, SymSpell normalization, retraining loop; Flask + WSGI serving on AWS ECS
- LR Consulting Maroc: confidential PDF‑to‑JSON API (Tabula/Pandas) with offline processing guarantees
- Wikreate Agency: React + Laravel data visualization interface with pagination and search; robust API integration and testing

## Technical Stack

```python
stack = {
    "serving": ["FastAPI", "BentoML", "vLLM", "LiteLLM"],
    "orchestration": ["Docker", "Docker Compose", "Kubernetes"],
    "tracking": ["MLflow", "DVC"],
    "monitoring": ["custom KPIs", "drift metrics (KS)", "SLO dashboards"],
    "infrastructure": ["AWS ECS/S3", "GCP", "Docker", "Kubernetes"],
    "messaging": ["MQTT"],
    "data_storage": ["S3/MinIO", "PostgreSQL", "MongoDB", "Redis"],
    "automation": ["CI/CD", "GitHub Actions"],
    "fundamentals": ["Adaptive batching", "Drift detection", "System design"]
}
```

## Writing

I write about practical ML system challenges with math and implementation:
- Time series analysis (ACF/PACF, Kalman filtering, spectral methods)
- Data drift (covariate shift, prior probability shift, sample selection bias)
- Inference optimization (batching strategies, processing patterns)
- CNNs and feature engineering in production contexts

---

## Contact

**GitHub:** [@IbLahlou](https://github.com/IbLahlou)  
**LinkedIn:** [ibrahimlahlou-ild01](https://www.linkedin.com/in/ibrahimlahlou-ild01)  
**Kaggle:** [ibrahimld01](https://www.kaggle.com/ibrahimld01)  
**Twitter/X:** [@ILoDo01](https://twitter.com/ILoDo01)  
**Email:** ibrahimlahlou021@gmail.com



