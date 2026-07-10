# **🏭 ISARM: Intelligent Shopfloor Allocation & Resiliency Mesh**

**Tata Steel Internship Project** — End-to-end Cyber-Physical Systems ML Pipeline  
Autoencoder-based Flight Risk Detection \+ Multi-Agent DQN Human-Cobot Allocation \+ Interactive Dashboard.

## **📊 Project Overview**

| Layer | Technology | Purpose |
| :---- | :---- | :---- |
| **Macro (Workforce)** | Deep Autoencoder (PyTorch) | Unsupervised anomaly detection on 30-dim employee profiles → Flight Risk Score \+ 16-dim Latent Vector |
| **Micro (Shopfloor)** | Multi-Head DQN (Gymnasium) | Real-time task allocation for 20 Humans \+ 5 Cobots \+ 2 Cranes across 3 Rolling Mills |
| **Fusion** | Latent Injection | Autoencoder latent vectors \+ risk scores fed into RL State Space (![][image1]) |
| **Visualization** | Plotly \+ NetworkX \+ Matplotlib | Topological Knowledge Graph (Risk-weighted) \+ Spatial Mesh (Real-time coords) |
| **Deployment** | Streamlit \+ Docker \+ K8s | Interactive Dashboard, Model Cards, Inference API |

## **📂 Repository Structure**

isarm/  
├── app/                             \# Streamlit frontend assets (future)  
├── data/  
│   ├── raw/                         \# Employee\_Feature\_Matrix.csv (generated)  
│   ├── processed/                   \# split\_data.npz (train/val tensors)  
│   ├── mock\_telemetry/              \# shopfloor\_state\_stream.csv (generated)  
├── models/  
│   ├── scalers/                     \# Individual fitted scalers (audit/ONNX)  
│   ├── autoencoder\_best.pth  
│   ├── dqn\_shopfloor.pth  
│   ├── preprocessor.joblib  
│   ├── threshold.json  
│   └── feature\_metadata.json  
├── outputs/  
│   ├── flight\_risk\_scores.csv       \# Primary deliverable (worker\_id, risk\_score, tier, latent)  
│   ├── latent\_vectors.npy           \# (500, 16\) for RL injection  
│   └── plots/                       \# topology\_workforce.png, spatial\_mesh.png, training\_curve.png  
├── notebooks/                       \# EDA / Prototyping (optional)  
├── src/  
│   ├── generate\_data.py             \# Day 1: Synthetic Data \+ Preprocessing  
│   ├── train\_autoencoder.py         \# Day 2: Autoencoder \+ Flight Risk Scoring  
│   ├── train\_rl.py                  \# Days 3-4: Gymnasium Env \+ DQN \+ Visualizations  
│   ├── app.py                       \# Day 5: Streamlit Dashboard  
│   └── populate\_artifact\_folders.py \# Helper: extracts scalers \+ generates telemetry  
├── Dockerfile  
├── requirements.txt  
└── README.md

## **⚙️ Prerequisites**

| Tool | Version | Install Guide |
| :---- | :---- | :---- |
| **Python** | 3.10 – 3.12 | [python.org](https://www.python.org/) / conda / pyenv |
| **Git** | ≥ 2.30 | [git-scm.com](https://git-scm.com/) |
| **Docker** | ≥ 24.0 | [docker.com](https://www.docker.com/) (for containerized deployment) |
| **NVIDIA GPU** | Optional | CUDA 11.8+ for accelerated training (CPU works, slower) |

## **🚀 Quick Start (Local)**

**1\. Clone & Enter**  
git clone \[https://github.com/\](https://github.com/)\<your-org\>/isarm.git  
cd isarm

**2\. Create Virtual Environment**  
python \-m venv .venv  
source .venv/bin/activate      \# Linux/macOS  
\# .venv\\Scripts\\activate       \# Windows PowerShell

**3\. Install Dependencies**  
pip install \--upgrade pip  
pip install \-r requirements.txt

**Note for GPU Support (Linux/WSL2):** \> Run pip install torch \--index-url https://download.pytorch.org/whl/cu118 (Auto-detected in code via torch.cuda.is\_available()).

## **🧠 Execution Pipeline (3 Blocks)**

### **Block 1: Data Generation & Autoencoder (Days 1-2)**

\# 1A. Generate synthetic workforce data \+ fit preprocessor  
python src/generate\_data.py  
\# ✅ Outputs: data/raw/Employee\_Feature\_Matrix.csv, models/preprocessor.joblib, data/processed/split\_data.npz

\# 1B. Train Autoencoder \+ Compute Flight Risk Scores  
python src/train\_autoencoder.py  
\# ✅ Outputs: models/autoencoder\_best.pth, models/threshold.json, outputs/flight\_risk\_scores.csv, outputs/latent\_vectors.npy

*Verify: Console shows Classification Report with Precision/Recall for Anomaly types.*

### **Block 2: Reinforcement Learning \+ Visualizations (Days 3-4)**

python src/train\_rl.py  
\# ✅ Outputs: models/dqn\_shopfloor.pth, outputs/plots/topology\_workforce.png, spatial\_mesh.png, training\_curve.png

*Runtime: \~30 min (CPU) / \~2 min (GPU) for 2000 episodes.* *Verify: Final AvgReward(100) \> 300 (converged from \-2500).*

### **Block 3: Populate Artifact Folders (MLOps Ready)**

python src/populate\_artifact\_folders.py  
\# ✅ Fills: models/scalers/ (individual scalers), data/mock\_telemetry/shopfloor\_state\_stream.csv

### **Block 4: Launch Dashboard (Day 5\)**

streamlit run src/app.py

*Opens:* http://localhost:8501 *Tabs:* Risk Radar | Topology Graph (Dark) | Spatial Mesh | RL Policy Tester | Model Cards

## **🐳 Docker Deployment (Production)**

**Build Image**  
docker build \-t isarm:latest .

**Run Container (CPU)**  
docker run \-d \-p 8501:8501 \--name isarm-dashboard isarm:latest

**Run Container (GPU \- NVIDIA)**  
docker run \-d \-p 8501:8501 \--gpus all \--name isarm-gpu isarm:latest

**Access:** http://\<host-ip\>:8501  
**Kubernetes (GKE/EKS/AKS)**  
kubectl apply \-f k8s-deployment.yaml   \# See Model Cards tab in Dashboard for YAML

## **📦 Key Artifacts & Metrics**

| Artifact | Path | Description |
| :---- | :---- | :---- |
| **Flight Risk Scores** | outputs/flight\_risk\_scores.csv | 500 workers × {worker\_id, reconstruction\_mse, flight\_risk\_score, risk\_tier, anomaly\_ground\_truth, latent\_vector} |
| **Latent Vectors** | outputs/latent\_vectors.npy | (500, 16\) numpy array — injected into RL State Space |
| **Autoencoder Threshold** | models/threshold.json | MSE cutoff (95th %ile Train Normal) |
| **DQN Policy** | models/dqn\_shopfloor.pth | 27-head Q-Network (Shared Backbone 424→128→128) |
| **Individual Scalers** | models/scalers/\*.joblib | Audit/ONNX export ready |
| **Telemetry Stream** | data/mock\_telemetry/shopfloor\_state\_stream.csv | 10k timesteps × 85 features (simulated sensor feed) |

### **Autoencoder Performance (Validation)**

| Class | Precision | Recall | F1 Score |
| :---- | :---- | :---- | :---- |
| **Normal** | 0.98 | 0.99 | 0.98 |
| **Anomaly (All Types)** | 0.92 | 0.85 | 0.88 |

**DQN Convergence:** Episode Reward \-2500 → \+422 (2000 eps, γ=0.99).

## **🧪 Testing & Validation**

Quick sanity check to load models and run one inference:  
python \-c "  
import torch, joblib, numpy as np  
pre \= joblib.load('models/preprocessor.joblib')  
ae \= torch.load('models/autoencoder\_best.pth', map\_location='cpu')  
dqn \= torch.load('models/dqn\_shopfloor.pth', map\_location='cpu')  
print('✅ All models load successfully')  
print(f'AE Params: {sum(p.numel() for p in ae.values()):,}')  
print(f'DQN Params: {sum(p.numel() for p in dqn.values()):,}')  
"

## **⚠️ Troubleshooting**

| Error | Fix |
| :---- | :---- |
| ModuleNotFoundError: torch | Run pip install torch (or CUDA variant as shown in Quick Start) |
| RuntimeError: CUDA out of memory | Reduce batch\_size in train\_autoencoder.py / train\_rl.py, or set device="cpu" in CFG |
| ValueError: shape mismatch (424 vs 445\) | Ensure load\_dqn(state\_dim=424, ...) in app.py matches train\_rl.py STATE\_DIM |
| st.plotly\_chart shows code not chart | Run pip install \--upgrade plotly streamlit and restart Streamlit |
| AttributeError: 'Styler' object has no attribute 'applymap' | Pandas ≥ 2.1: use .style.map() (already fixed in app.py) |
| Dashboard blank / 500 error | Check streamlit run src/app.py logs; verify outputs/flight\_risk\_scores.csv exists |

## **📋 Requirements (requirements.txt)**

streamlit\>=1.35  
pandas\>=2.1  
numpy\>=1.24  
scikit-learn\>=1.3  
torch\>=2.3  
gymnasium\>=0.29  
networkx\>=3.2  
plotly\>=5.20  
matplotlib\>=3.8  
tqdm\>=4.66  
joblib\>=1.3

## **📇 Model Cards (Auto-generated in Dashboard)**

* **Autoencoder:** Architecture, Threshold, Anomaly Types, Training Hyperparams  
* **DQN Agent:** MDP Formulation, Reward Function, Multi-Head Architecture, Convergence Plot  
* **Deployment:** Dockerfile, K8s YAML, Resource Limits, Health Checks

## **📜 Author & License**

**Principal Research Scientist Architecture — Cyber-Physical Systems** *Intern: nichette — Tata Steel ISARM Project*  
**License:** Proprietary (Tata Steel Internal Use) — Do not distribute externally without approval.

### **⚡ Quick Commands Cheatsheet**

**Full rebuild from scratch:**  
rm \-rf data/processed models/\*.pth models/\*.joblib outputs/ && \\  
python src/generate\_data.py && \\  
python src/train\_autoencoder.py && \\  
python src/train\_rl.py && \\  
python src/populate\_artifact\_folders.py && \\  
streamlit run src/app.py  


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAWCAYAAAAmaHdCAAABFklEQVR4Xu3SPUtCURjA8ScIi4ZsycUGv0AvuNQgQTQKDq31DRwCFxdt7GUVwWitvkBgQWRbIA6GENjS7i4UgVT/4znlcw/e0EmC/vAb7nOuh3vPVeS/v9c2btBAHVUs4woxdV9oRTxjRc3W0UVLzUJL4ROr/gJd4tgfDusUH5jzF6iMTX84LHOjeZJbbElwsyim1PWvzWMPZ2iL3bQnwVecxa66/ins1I/EbrSvZju4V9f9FlHzh64NsZuYH35XEvsVA5nHf/SHrkO8IIIELvAq9j+TG9wmco53FGRwmNPIooOkm5kW8IYZNet3jSUcoIknp+LmujQevNnYnYg9bFNcL4zTHTJYk5DPPErmnMwZ5v2FyfcFhN8uTPR2Vu4AAAAASUVORK5CYII=>