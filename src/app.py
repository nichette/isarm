# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from pathlib import Path
import sys

# --- PATH SETUP ---
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

# --- CONFIG ---
CFG = {
    "paths": {
        "models": ROOT / "models",
        "outputs": ROOT / "outputs",
        "raw": ROOT / "data/raw",
        "plots": ROOT / "outputs/plots",
    },
    "device": "cpu",  # Dashboard runs on CPU
}


# ==============================================================================
# 1. LOAD ARTIFACTS (Cached)
# ==============================================================================
@st.cache_resource
def load_preprocessor():
    return joblib.load(CFG["paths"]["models"] / "preprocessor.joblib")


@st.cache_resource
def load_autoencoder(input_dim, bottleneck=16):
    class DeepAutoencoder(nn.Module):
        def __init__(self, input_dim, bottleneck_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, bottleneck_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    model = DeepAutoencoder(input_dim, bottleneck)
    model.load_state_dict(
        torch.load(CFG["paths"]["models"] / "autoencoder_best.pth", map_location="cpu")
    )
    model.eval()
    return model


@st.cache_resource
def load_dqn(state_dim, action_dim, n_entities):
    class QNetwork(nn.Module):
        def __init__(self, state_dim, action_dim_per_entity, n_entities):
            super().__init__()
            self.n_entities = n_entities
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
            )
            self.heads = nn.ModuleList(
                [nn.Linear(128, action_dim_per_entity) for _ in range(n_entities)]
            )

        def forward(self, x):
            feat = self.net(x)
            return torch.stack([h(feat) for h in self.heads], dim=1)

    model = QNetwork(state_dim, action_dim, n_entities)
    model.load_state_dict(
        torch.load(CFG["paths"]["models"] / "dqn_shopfloor.pth", map_location="cpu")
    )
    model.eval()
    return model


@st.cache_data
def load_data():
    df_risk = pd.read_csv(CFG["paths"]["outputs"] / "flight_risk_scores.csv")
    df_risk["latent_vector"] = df_risk["latent_vector"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
    )
    df_raw = pd.read_csv(CFG["paths"]["raw"] / "Employee_Feature_Matrix.csv")
    with open(CFG["paths"]["models"] / "threshold.json") as f:
        thresh = json.load(f)["threshold_mse"]
    with open(CFG["paths"]["models"] / "feature_metadata.json") as f:
        meta = json.load(f)
    return df_risk, df_raw, thresh, meta


# Load everything
pre = load_preprocessor()
df_risk, df_raw, THRESHOLD, META = load_data()
INPUT_DIM = pre.transform(
    df_raw.drop(columns=["worker_id", "is_anomaly", "anomaly_type"])
).shape[1]
ae = load_autoencoder(INPUT_DIM)
dqn = load_dqn(
    state_dim=424, action_dim=6, n_entities=27
)  # Must match train_rl.py dims

# ==============================================================================
# 2. UI LAYOUT
# ==============================================================================
st.set_page_config(page_title="ISARM - Tata Steel", layout="wide", page_icon="🏭")
st.title("🏭 ISARM: Intelligent Shopfloor Allocation & Resiliency Mesh")
st.caption(
    "Tata Steel Internship Project | Autoencoder Flight Risk + DQN Human-Cobot Allocation"
)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    st.metric("Workers Monitored", len(df_risk))
    st.metric("High Risk (Critical)", len(df_risk[df_risk["risk_tier"] == "Critical"]))
    st.metric("Autoencoder Threshold (MSE)", f"{THRESHOLD:.4f}")
    st.divider()
    st.subheader("Models Loaded")
    st.success("✅ Autoencoder (Flight Risk)")
    st.success("✅ DQN Agent (Shopfloor Allocation)")
    st.success("✅ Preprocessor (Feature Pipeline)")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Workforce Risk Radar",
        "🕸️ Topology Graph",
        "🗺️ Spatial Mesh",
        "🤖 RL Policy Tester",
        "📋 Model Cards & Docs",
    ]
)

# ==============================================================================
# TAB 1: WORKFORCE RISK RADAR
# ==============================================================================
with tab1:
    st.subheader("Employee Flight Risk Scores")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tier_filter = st.multiselect(
            "Risk Tier",
            ["Low", "Medium", "Critical"],
            default=["Low", "Medium", "Critical"],
        )
    with col_f2:
        type_filter = st.multiselect(
            "Anomaly Type",
            df_risk["anomaly_ground_truth"].unique(),
            default=df_risk["anomaly_ground_truth"].unique(),
        )
    with col_f3:
        search = st.text_input("Search Worker ID")

    df_show = df_risk[
        (df_risk["risk_tier"].isin(tier_filter))
        & (df_risk["anomaly_ground_truth"].isin(type_filter))
    ]
    if search:
        df_show = df_show[df_show["worker_id"].str.contains(search, case=False)]

    # Color mapping
    def color_tier(v):
        if v == "Critical":
            return "background-color: #ff4b4b"
        if v == "Medium":
            return "background-color: #ffa500"
        return "background-color: #00cc96"

    st.dataframe(
        df_show[
            [
                "worker_id",
                "reconstruction_mse",
                "flight_risk_score",
                "risk_tier",
                "anomaly_ground_truth",
            ]
        ]
        .style.map(color_tier, subset=["risk_tier"])
        .format({"reconstruction_mse": "{:.4f}", "flight_risk_score": "{:.3f}"}),
        width="stretch",
        height=500,
    )

    # Histogram
    fig = px.histogram(
        df_risk,
        x="flight_risk_score",
        color="risk_tier",
        nbins=50,
        title="Flight Risk Score Distribution",
        color_discrete_map={
            "Low": "#00cc96",
            "Medium": "#ffa500",
            "Critical": "#ff4b4b",
        },
    )
    fig.add_vline(
        x=0.7, line_dash="dash", line_color="red", annotation_text="Critical Threshold"
    )
    st.plotly_chart(fig, width="stretch")


# ==============================================================================
# TAB 2: TOPOLOGY GRAPH (Interactive Plotly - Dark Theme)
# ==============================================================================
with tab2:
    st.subheader("Workforce Knowledge Topology")
    st.caption(
        "Nodes = Workers (size/risk) + Skills. Edges = Worker-Skill Assignment. Red = High Flight Risk."
    )

    # Rebuild graph (cached)
    @st.cache_data
    def build_graph(_df_risk, _df_raw):
        G = nx.Graph()
        active_ids = _df_risk["worker_id"].tolist()
        risk_map = dict(zip(_df_risk["worker_id"], _df_risk["flight_risk_score"]))
        for wid in active_ids:
            G.add_node(wid, risk=risk_map[wid], type="human")
        for skill in _df_raw["primary_skill_cluster"].unique():
            G.add_node(f"Skill:{skill}", type="skill", risk=0)
        for _, row in _df_raw[_df_raw["worker_id"].isin(active_ids)].iterrows():
            G.add_edge(row["worker_id"], f"Skill:{row['primary_skill_cluster']}")
        return G

    G = build_graph(df_risk, df_raw)
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

    # --- EXTRACT COORDINATES & ATTRIBUTES (REQUIRED FOR PLOTLY) ---
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        if G.nodes[n]["type"] == "human":
            r = G.nodes[n]["risk"]
            node_color.append(r)
            node_size.append(10 + r * 40)
            node_text.append(f"{n}<br>Risk: {r:.2f}")
        else:
            node_color.append(0)
            node_size.append(15)
            node_text.append(n)

    # --- PLOTLY FIGURE (DARK THEME) ---
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#444"),
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Flight Risk"),
                cmin=0,
                cmax=1,
            ),
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Interactive Workforce Topology",
        showlegend=False,
        hovermode="closest",
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch", theme=None, key="topology_graph")

# ==============================================================================
# TAB 3: SPATIAL MESH (Static Image from train_rl)
# ==============================================================================
with tab3:
    st.subheader("Shopfloor Spatial Mesh (Human-Cobot-Crane Positions)")
    st.caption(
        "Final state of trained DQN episode. Red = High Risk Human. Blue = Cobot. Green = Crane."
    )

    img_path = CFG["paths"]["plots"] / "spatial_mesh.png"
    if img_path.exists():
        st.image(str(img_path), width="stretch")
    else:
        st.warning("Run `train_rl.py` first to generate spatial_mesh.png")

    # Also show training curve
    curve_path = CFG["paths"]["plots"] / "training_curve.png"
    if curve_path.exists():
        st.image(
            str(curve_path),
            caption="DQN Training Convergence (Rolling Avg 50)",
            width="stretch",
        )

# ==============================================================================
# TAB 4: RL POLICY TESTER (Inference)
# ==============================================================================
with tab4:
    st.subheader("🤖 DQN Policy Inference Engine")
    st.caption(
        "Input a shopfloor state vector → Get optimal task assignment for 27 entities."
    )

    st.info(
        "State Dim: **424** (3 Mills + 6 cranes + 15 Cobots + 400 Human Features). Auto-filled with random valid state for demo."
    )

    if st.button("🎲 Generate Random State & Predict"):
        # Build a valid dummy state matching ShopfloorEnv._get_obs()
        # 3 mills + 2*3 cranes + 5*3 cobots + 20*20 humans = 3+6+15+400 = 424?
        # train_rl.py STATE_DIM = 3 + 6 + 15 + 400 = 424.
        # But QNetwork init used 45+400=445. Mismatch!
        # We'll use the actual STATE_DIM from train_rl: 424.
        # For demo, just random.
        dummy_state = np.random.randn(424).astype(
            np.float32
        )  # Match train_rl STATE_DIM

        with torch.no_grad():
            state_t = torch.FloatTensor(dummy_state).unsqueeze(0)
            q_vals = dqn(state_t)  # (1, 27, 6)
            actions = q_vals.argmax(dim=2).squeeze(0).numpy()

        task_names = ["Idle", "Mill1", "Mill2", "Mill3", "Scrap", "Maint"]
        df_act = pd.DataFrame(
            {
                "Entity": [f"Human_{i}" for i in range(20)]
                + [f"Cobot_{i}" for i in range(5)]
                + [f"Crane_{i}" for i in range(2)],
                "Assigned_Task": [task_names[a] for a in actions],
                "Task_ID": actions,
            }
        )
        st.dataframe(df_act, width="stretch")

        # Show Q-values for first human
        fig = px.bar(
            x=task_names, y=q_vals[0, 0, :].numpy(), title="Q-Values for Human_0"
        )
        st.plotly_chart(fig, width="stretch")

# ==============================================================================
# TAB 5: MODEL CARDS & DOCS
# ==============================================================================
with tab5:
    st.subheader("📋 Model Cards & Deployment Blueprint")

    with st.expander("📄 Autoencoder Model Card", expanded=True):
        st.markdown(f"""
        **Model:** Deep Autoencoder (Anomaly Detection)
        **Task:** Unsupervised Flight Risk Detection
        **Architecture:** Input({INPUT_DIM}) → 64 → 32 → **Bottleneck(16)** → 32 → 64 → Output({INPUT_DIM})
        **Activation:** ReLU + BatchNorm1d + Dropout(0.2/0.1)
        **Loss:** MSE (Reconstruction)
        **Threshold:** {THRESHOLD:.6f} (95th %ile Train Normal MSE)
        **Training:** 100 Epochs, AdamW(lr=1e-3, wd=1e-5), Batch=64
        **Data:** 500 Workers, 30 Raw Features → 45 Processed (OneHot expanded)
        **Anomaly Types Injected:** Silent Quitter (10), SPOF (10), Burnout (10)
        **Artifacts:** `models/autoencoder_best.pth`, `models/preprocessor.joblib`, `models/threshold.json`
        """)

    with st.expander("🤖 DQN Agent Model Card"):
        st.markdown("""
        **Model:** Multi-Head DQN (Independent Q-Learning)
        **Task:** Human-Cobot-Crane Task Allocation (MDP)
        **State Space:** Shopfloor Backlogs + Entity Poses + **Human Latent Vectors(16) + Risk Scores(1)**
        **Action Space:** MultiDiscrete([6] × 27) → 6 Tasks × 27 Entities (20H+5C+2Cr)
        **Reward:** $R_t = \\alpha\\cdot Throughput - \\beta\\cdot Fatigue - \\gamma\\cdot Risk$
        **Architecture:** Shared Backbone(424→128→128) + 27 Independent Heads(128→6)
        **Training:** 2000 Episodes, 200 Steps, γ=0.99, ε-greedy(1.0→0.05), Buffer=50k, Batch=64
        **Convergence:** Avg Reward -2500 → +422
        **Artifacts:** `models/dqn_shopfloor.pth`
        """)

    with st.expander("🐳 Deployment Blueprint (Docker + K8s)"):
        st.code(
            """
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
COPY outputs/ ./outputs/
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
        """,
            language="dockerfile",
        )
        st.code(
            """
# k8s-deployment.yaml
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
    metadata:
      labels:
        app: isarm
    spec:
      containers:
      - name: dashboard
        image: your-registry/isarm:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: isarm-svc
spec:
  selector:
    app: isarm
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
        """,
            language="yaml",
        )

    with st.expander("📊 Data Dictionary"):
        st.json(META)

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.caption(
    "ISARM v1.0 | Tata Steel Internship"
)
