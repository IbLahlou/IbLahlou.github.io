---
layout: page
icon: fas fa-briefcase
order: 6
---

<link rel="icon" type="image/x-icon" href="{{ '/assets/img/favicons/ild01@4x.ico' | relative_url }}">

## Projects

### Forensic Image Feature Matching — DICE | DATALAB
- Scope: designed and operated a production model‑serving app for forensic image feature matching with online quality monitoring
- Stack: FastAPI/BentoML, Docker/Docker Compose, MLflow, KS‑based drift detection, separate ML workers and web workers for concurrency; Kubernetes/ECS‑ready
- Key challenges solved:
  - Scalable concurrency: isolated ML workers for heavy inference, web workers for HTTP handling; tunable worker pools and adaptive batching
  - Reproducibility: MLflow‑tracked runs with pinned data/model/code; deterministic preprocessing
  - Online monitoring: KS scores on match distributions and response latency; automated alerts and rollback strategies
  - Retrainable workflows: templated pipelines for data updates and model refresh with CI/CD gates
- Value: improved robustness under real‑world distribution shift, predictable latency, and safer deployments via progressive delivery

### Smart NLP Pipeline (AWS ECS) — 3D Smart Factory
- Built an end‑to‑end text processing pipeline using AWS S3 for storage and SymSpell for normalization and retraining
- Served predictions via Flask + WSGI; containerized and deployed on AWS ECS with production configuration
- Introduced data quality checks and automated QoS thresholds across multiple iterations
- Value: reliable retraining loop with traceable artifacts and stable inference in production

### Confidential API Data Pipeline — LR Consulting Maroc
- Implemented secure, offline PDF‑to‑JSON processing using Python Tabula and Pandas; avoided cloud usage to preserve confidentiality
- Exposed well‑structured JSON for downstream HTTP integrations; enforced filtering and formatting contracts
- Value: efficient automation with compliance constraints; maintainable API interfaces

### Full‑Stack Data Visualization — Wikreate Agency
- Developed a dynamic interface with advanced table pagination and search from Figma designs
- React frontend with DataTables/jQuery for interaction; Laravel/PHP backend with RESTful endpoints, authentication, and Axios integration
- Comprehensive API testing in Postman; performance tuning and responsive design
- Value: production‑grade UI patterns applicable to ML monitoring dashboards and admin consoles

### Reusable Templates & Documentation — DICE | DATALAB
- Authored template code for model serving and workflow orchestration targeting retrainable models
- Documented operational practices: versioning, CI/CD, monitoring, drift detection, and rollback procedures
- Value: accelerates team velocity and improves reliability across ML services
