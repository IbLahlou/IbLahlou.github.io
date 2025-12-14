---
layout: page
icon: fas fa-briefcase
order: 6
---

<link rel="icon" type="image/x-icon" href="{{ '/assets/img/favicons/ild01@4x.ico' | relative_url }}">

## Projects

### DialFlow — Gen‑AI Voice Agent (Winner, 2025)
- Award: 1st place, First Moroccan Gen‑AI Voice Bot Hackathon by AI Crafters @ Technopark (MITC), Casablanca
- Mission: customizable AI voice agent for multilingual customer service with task automation and analytics
- Stack: Twilio (voice/telephony), FastAPI (real‑time API), Redis (state/caching/queues), ElevenLabs (TTS), LangChain (prompt/flow orchestration)
- Architecture:
  - Voice call ingestion via Twilio; secure webhooks and call session management
  - Streaming pipeline: FastAPI WebSocket endpoints with back‑pressure; worker isolation for TTS/STT tasks
  - State and rate control via Redis (session state, throttling, caching of prompt/tool results)
  - Orchestrated dialogs using LangChain tools/agents; deterministic prompt versioning and audit
  - Observability: latency percentiles, error budgets, call completion funnels, per‑intent success metrics
- Reliability & scale: adaptive batching for synthesis, retry policies and circuit breakers, canary releases for new prompt versions
- Value: production‑ready voice assistant foundation with measurable service KPIs and cost controls; accelerates integration for Moroccan businesses

### Forensic Image Feature Matching — DICE | DATALAB
- Scope: designed and operated a production model‑serving app for forensic image feature matching with online quality monitoring
- Stack: FastAPI/BentoML, Docker/Docker Compose, MLflow, KS‑based drift detection, separate ML workers and web workers for concurrency; Kubernetes/ECS‑ready
- Key challenges solved:
  - Scalable concurrency: isolated ML workers for heavy inference, web workers for HTTP handling; tunable worker pools and adaptive batching
  - Reproducibility: MLflow‑tracked runs with pinned data/model/code; deterministic preprocessing
  - Online monitoring: KS scores on match distributions and response latency; automated alerts and rollback strategies
  - Retrainable workflows: templated pipelines for data updates and model refresh with CI/CD gates
- Value: improved robustness under real‑world distribution shift, predictable latency, and safer deployments via progressive delivery

### Allam LLM Educational Platform — Atlas Innovators (Finals, Riyadh 2024)
- Event: Allam Challenge 2024 Finals, Riyadh (Sep 7–10); team Atlas Innovators — Othman Moussaoui, Ismail Hamdach, Hicham Maghraoui, Ibrahim Lahlou
- Mission: interactive Arabic learning platform for children; enable Allam LLM to “see”, “hear”, “speak”, and “generate images”
- Backend role:
  - FastAPI service exposing clean endpoints for model inference and multimodal interactions
  - Child/parent management: entity modeling, authentication/authorization, and session handling
  - MySQL storage for profiles and insights (engagement, progress, comprehension metrics) to personalize learning
  - Integration as the final stage: endpoint governance, schema contracts, and reliability checks before demo
- Architecture: modular services for perception (vision/audio), TTS/ASR, and LLM reasoning; deterministic prompt versions and telemetry for education KPIs
  - Components:
    - Vector store and document embeddings for semantic search over educational content
    - LLM serving backend with Allam and open‑source models: Whisper Large v3 (ASR), XTTS (TTS), FLUX.1 (text‑to‑image), vision modules
    - Providers for inference acceleration and model variety (e.g., Groq)
    - Database: MySQL for child/parent profiles and insight metrics
    - Profiling backend: Node.js and FastAPI integration for endpoint management and telemetry
    - Frontend: Axios and fetch bridge to modules (Story, Chatbot, Parent Interface, Child Interface, Authentication, Quiz)
- Value: production‑aligned backend foundation that supports safe, measurable, and adaptive learning experiences for kids

### Smart NLP Pipeline (AWS ECS) — 3D Smart Factory
- Built an end‑to‑end text processing pipeline using AWS S3 for storage and SymSpell for normalization and retraining
- Served predictions via Flask + WSGI; containerized and deployed on AWS ECS with production configuration
- Introduced data quality checks and automated QoS thresholds across multiple iterations
- Value: reliable retraining loop with traceable artifacts and stable inference in production

### OCR API Data Pipeline — LR Consulting Maroc
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
