# ISARM Project — Reference Library Index
**Tata Steel ISARM: Intelligent Shopfloor Allocation & Resiliency Mesh**
**Compiled by:** Principal Research Scientist, Cyber-Physical Systems
**Date:** 2026-07-11
**Classification:** Internal — Tata Steel R&D

---

## 📚 Reference Catalogue (10 Core Sources)

| # | File | Source | Domain | Key Concepts Applied |
|---|------|--------|--------|---------------------|
| 01 | `01_deep_learning_autoencoders.md` | Goodfellow, Bengio, Courville — *Deep Learning* (Ch. 14) | Deep Learning Theory | Autoencoder architecture, bottleneck representation, reconstruction error as anomaly score, regularized autoencoders |
| 02 | `02_rl_sutton_barto.md` | Sutton & Barto — *Reinforcement Learning: An Introduction* (Ch. 3, 4, 6, 11) | RL Theory | MDP formulation, value iteration, DQN, experience replay, target networks, exploration-exploitation |
| 03 | `03_sklearn_preprocessing.md` | Scikit-Learn User Guide — *Preprocessing Data* | Feature Engineering | ColumnTransformer, FunctionTransformer, OneHotEncoder, StandardScaler/MinMaxScaler, pipeline composition |
| 04 | `04_pytorch_docs.md` | PyTorch Documentation — *nn.Module, MSELoss, DataLoader, optim* | Implementation | Module design, loss functions, DataLoader batching, AdamW optimizer, gradient clipping |
| 05 | `05_gymnasium_env.md` | Farama Foundation Gymnasium — *Custom Environments* | RL Engineering | Env API (reset/step), observation/action spaces, MultiDiscrete, rendering, vectorization |
| 06 | `06_industrial_anomaly_survey.md` | Chalapathy & Chawla — *Deep Learning for Anomaly Detection: A Survey* (2019) | Industrial ML | Contextual anomalies, reconstruction-based detection, threshold selection, evaluation metrics |
| 07 | `07_manufacturing_fe_survey.md` | *Feature Engineering for ML in Manufacturing: A Review* (J. Mfg. Systems, 2022) | Manufacturing Analytics | Temporal features, rolling windows, entropy measures, cross-training metrics, digital twin telemetry |
| 08 | `08_networkx_visualization.md` | NetworkX Documentation + D3.js Force-Directed Theory | Graph Visualization | Spring layout, node attributes, edge weighting, bipartite graphs (workers↔skills), Plotly integration |
| 09 | `09_streamlit_deployment.md` | Streamlit Docs — *Deployment, Caching, Session State* | MLOps/UI | st.cache_resource, st.cache_data, components.v1.html, Plotly charts, multipage apps |
| 10 | `10_mlops_model_cards.md` | Google Model Card Toolkit + *MLOps: Continuous Delivery for ML* | Governance | Model cards, data cards, drift monitoring, Docker/K8s deployment, health checks |

---

## 🔗 Cross-Cutting Themes

| Theme | References |
|-------|------------|
| **Anomaly Detection → RL State Fusion** | 01, 02, 06 |
| **Heterogeneous Feature Engineering** | 03, 07 |
| **Production RL (Sim-to-Real)** | 02, 05, 10 |
| **Visualization for Ops** | 08, 09 |
| **Reproducibility & Governance** | 04, 09, 10 |

---

## 📖 Usage Guide
- Each file contains **verbatim snippets** (fair use: research/education) with **section markers** linking to ISARM components.
- Use `grep -r "ISARM_MAP:" references/` to find all mappings.
- For defense: cite the **original source**, not this compilation.
