# Reference 10: MLOps — Model Cards & Deployment
**Sources:**
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAT* '19.
- Google Cloud. *Model Card Toolkit*. https://github.com/tensorflow/model-card-toolkit
- Breuel, T. (2020). *MLOps: Continuous Delivery for ML*. O'Reilly.
- Kubernetes Documentation: *Deployments, Services, Health Checks*.

---

## ISARM_MAP: [Model Cards + Deployment] → `src/app.py` Tab 5 + `Dockerfile` + `k8s-deployment.yaml`

### Model Card Sections (Mitchell et al.)

#### 1. Model Details
> "Basic information: name, version, type, architecture, training data, license."

**ISARM Autoencoder Card (Tab 5):**

Model: Deep Autoencoder (Anomaly Detection)

Task: Unsupervised Flight Risk Detection

Architecture: 45→64→32→16→32→64→45 (ReLU+BN+Dropout)

Training: 500 workers, 100 epochs, AdamW(lr=1e-3)

Threshold: 95th %ile Train Normal MSE

Artifacts: autoencoder_best.pth, preprocessor.joblib, threshold.json

#### 2. Intended Use
> "Primary use cases, users, out-of-scope uses."

**ISARM:**
- Primary: Tata Steel HR/Operations — daily flight risk scoring
- Users: Shift managers, workforce planners
- Out-of-scope: Hiring decisions, disciplinary actions (human-in-the-loop required)

#### 3. Metrics
> "Performance measures: accuracy, fairness, robustness... with confidence intervals."

**ISARM (Validation Set):**
| Metric | Normal | Anomaly |
|--------|--------|---------|
| Precision | 0.98 | 0.92 |
| Recall | 0.99 | 0.85 |
| F1 | 0.98 | 0.88 |

#### 4. Training Data
> "Data provenance, collection, preprocessing, splits."

**ISARM:**
- Synthetic: 500 workers, 30 features → 45 processed
- Anomaly injection: 6% (3 types × 10 each)
- Split: 80/20 stratified on anomaly_type
- Preprocessing: ColumnTransformer (log1p, Standard, MinMax, OneHot)

#### 5. Ethical Considerations
> "Bias, privacy, misuse potential... mitigation strategies."

**ISARM:**
- **Bias:** Synthetic data balanced across demographics (not modeled)
- **Privacy:** No real PII; worker_id = `W_XXXX`
- **Misuse:** Risk score = decision support ONLY; threshold explainable
- **Mitigation:** Human-in-the-loop; model card transparency

---

## DQN Agent Model Card

| Section | Content |
|---------|---------|
| **Model** | Multi-Head DQN (Shared Backbone 424→128→128 + 27×Heads 128→6) |
| **Task** | Human-Cobot-Crane Task Allocation (MDP) |
| **State** | 424-dim: backlogs(3) + cranes(6) + cobots(15) + humans(400) |
| **Action** | MultiDiscrete(6×27) — 6 tasks × 27 entities |
| **Reward** | `R = α·Throughput − β·Fatigue − γ·Risk` |
| **Training** | 2000 eps, 200 steps, γ=0.99, ε=1.0→0.05, Buffer=50k |
| **Convergence** | Avg Reward -2500 → +422 |
| **Artifacts** | `dqn_shopfloor.pth` |

---

## Data Card (Schema)

**ISARM `feature_metadata.json` captures:**

json

{

"raw_columns": [...],

"processed_dim": 45,

"column_groups": {...},

"anomaly_mapping": {0:"Normal", 1:"Silent_Quitter", 2:"SPOF", 3:"Burnout"}

}

Per Google Data Card principles: provenance, schema, intended use.

---

## Deployment Architecture

### Docker (Multi-stage not needed — slim base)

dockerfile

FROM python:3.11-slim
System deps for matplotlib

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ models/ outputs/ data/ ./

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

### Kubernetes (GKE/EKS/AKS Ready)

yaml
Deployment

apiVersion: apps/v1

kind: Deployment

metadata:

name: isarm-dashboard

spec:

replicas: 2

selector:

matchLabels:

app: isarm

template:

spec:

containers:

- name: dashboard

image: your-registry/isarm:latest

ports:

- containerPort: 8501

resources:

requests: {memory: "1Gi", cpu: "500m"}

limits: {memory: "2Gi", cpu: "1000m"}

livenessProbe:

httpGet: {path: /_stcore/health, port: 8501}

initialDelaySeconds: 30

periodSeconds: 10
Service

apiVersion: v1

kind: Service

metadata:

name: isarm-svc

spec:

selector: {app: isarm}

ports: [{port: 80, targetPort: 8501}]

type: LoadBalancer

### Monitoring & Drift Detection
| Check | Frequency | Method |
|-------|-----------|--------|
| Input drift | Daily | Compare `preprocessor.mean_` vs batch mean |
| Prediction drift | Weekly | Monitor risk score distribution shift |
| Model performance | Monthly | Labeled audit (if labels available) |
| Latency | Continuous | Prometheus + Grafana on `/metrics` |

---

## CI/CD Pipeline (GitLab/GitHub Actions)

yaml

stages:

- test: pytest src/ (unit tests for data gen, env)

- train: python src/generate_data.py && python src/train_autoencoder.py && python src/train_rl.py

- build: docker build -t REGISTRY/isarm:REGISTRY/isarm:CI_COMMIT_SHA .

- deploy: kubectl apply -f k8s/ (to staging)

- promote: manual approval → production

---

## Key MLOps Principles Applied

| Principle | ISARM Implementation |
|-----------|---------------------|
| **Reproducibility** | Fixed seeds, `requirements.txt`, synthetic data |
| **Traceability** | `feature_metadata.json`, `threshold.json`, model cards |
| **Monitoring** | Drift detection via scaler stats, health endpoint |
| **Rollback** | Versioned Docker images, K8s rollout undo |
| **Governance** | Model cards in UI, data cards in metadata |

