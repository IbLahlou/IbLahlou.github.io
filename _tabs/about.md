---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

<link rel="icon" type="image/x-icon" href="{{ '/assets/img/favicons/ild01@4x.ico' | relative_url }}">

## TL;DR

ML Engineer building production-grade systems. I turn research papers into scalable infrastructure—then watch them drift and fix them in production.

---

## The Long Version

I'm **Ibrahim Lahlou**, an ML Engineer who believes that **getting models to production is the easy part**. The hard part? Keeping them there.

My work sits at the intersection of machine learning and systems engineering, where I spend most of my time thinking about:

- **Why your model's accuracy dropped from 94% to 78% overnight** (spoiler: data drift)
- **How to serve 1000 req/s without melting your GPU** (batching strategies that actually work)
- **What happens when your training distribution looks nothing like production** (covariate shift is real)

### What I Do

**MLOps Engineering** — The unglamorous work that keeps ML systems alive in the wild:

```python
stack = {
    "serving": ["FastAPI", "BentoML", "Yatai", "Ray Serve"],
    "orchestration": ["Prefect", "Airflow"],
    "tracking": ["MLflow", "DVC"],
    "monitoring": ["Evidently AI", "Prometheus", "Grafana"],
    "infrastructure": ["GCP", "Docker", "Kubernetes"],
    "data_storage": ["MinIO", "MongoDB", "PostgreSQL", "Redis", "MySQL"],
    "automation": ["CI/CD", "GitHub Actions", "Selenium"],
    "fundamentals": ["Adaptive batching", "Drift detection", "System design"]
}
```

**Core Focus:** Building production ML systems with modern tools while understanding the underlying principles. I know *why* adaptive batching works, *when* to use stream vs. batch processing, and *how* statistical drift tests actually function—not just which framework to `pip install`.

### Background

**State Engineering Diploma** — Data Science & Cloud Computing
ENSA Oujda | UMP (2019-2024)

Key areas: ML systems, distributed computing, stochastic processes, cloud-native architecture.

**2 years** building MLOps pipelines with FastAPI, orchestrating workflows in Prefect, tracking experiments in MLflow, and deploying on GCP. I understand the theory behind adaptive batching and drift detection, then implement it with tools that work.

### Technical Writing

I write about the messy realities of production ML:

- **Time series fundamentals** — ACF/PACF, Kalman filtering, spectral methods
- **Data drift taxonomy** — Covariate shift vs. prior probability shift vs. sample selection bias
- **CNN architectures** — Convolution math, pooling strategies, implementation patterns
- **Inference optimization** — Batch/stream/microbatch tradeoffs, latency budgets

You'll find mathematical rigor, implementation details, and the kind of edge cases that only surface at 3 AM in production.

### Philosophy

**Code is static. Data evolves. Models decay.**

The best ML systems aren't the ones with the highest accuracy—they're the ones that *detect when they're wrong* and *recover gracefully*.

I design for observability first, performance second, and accuracy third. Because a model you can't monitor is a model you can't trust.

---

## Connect

- **GitHub:** [@IbLahlou](https://github.com/IbLahlou)
- **Twitter:** [@ILoDo01](https://twitter.com/ILoDo01)
- **Email:** ibrahimlahlou021@gmail.com

> "In theory, theory and practice are the same. In practice, they're not."
> — Every MLOps engineer ever



