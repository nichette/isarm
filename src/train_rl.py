# src/train_rl.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from collections import deque, namedtuple
import random
from pathlib import Path
import joblib
import matplotlib

matplotlib.use("Agg")  # No GUI backend needed
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================
CFG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "episodes": 2000,
    "max_steps": 200,  # Steps per episode
    "gamma": 0.99,
    "lr": 1e-3,
    "batch_size": 64,
    "buffer_size": 50000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay": 0.995,
    "target_update": 10,
    "hidden_dim": 128,
    "paths": {
        "models": "models",
        "outputs": "outputs",
        "raw": "data/raw",
        "plots": "outputs/plots",
    },
}
for p in CFG["paths"].values():
    Path(p).mkdir(parents=True, exist_ok=True)
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
rng = np.random.default_rng(CFG["seed"])
print(f"Device: {CFG['device']}")

# ==============================================================================
# 1. LOAD ARTIFACTS FROM BLOCK 1 (Autoencoder + Scaler + Risk Scores)
# ==============================================================================
print("[1/5] Loading Block 1 Artifacts...")
# Load Preprocessor
preprocessor = joblib.load(Path(CFG["paths"]["models"]) / "preprocessor.joblib")
# Load Flight Risk Scores (contains latent_vector per worker)
df_risk = pd.read_csv(Path(CFG["paths"]["outputs"]) / "flight_risk_scores.csv")
# Parse latent vectors stored as strings "[...]"
df_risk["latent_vector"] = df_risk["latent_vector"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ")
)
latent_dict = dict(
    zip(df_risk["worker_id"], df_risk["latent_vector"])
)  # worker_id -> (16,)
risk_dict = dict(
    zip(df_risk["worker_id"], df_risk["flight_risk_score"])
)  # worker_id -> float

# Load Autoencoder Architecture (to get input_dim if needed, but we use latent_dict directly)
# We need the input_dim for the RL State: Shopfloor_State + Worker_Latent_Vector
# Shopfloor state dim (defined below) + 16 (latent) + 1 (risk_score) = State Dim

# ==============================================================================
# 2. SHOPFLOOR SIMULATION (Internal Generator - Lightweight)
# ==============================================================================
# Entities: 3 Mill Lines, 2 Cranes, 5 Cobots, 20 Humans (subset of 500)
N_MILLS = 3
N_CRANES = 2
N_COBOTS = 5
N_HUMANS = 20
FACTORY_BOUNDS = {"x": (0, 200), "y": (0, 100)}

# Sample N_HUMANS workers from the 500 for this shift
active_worker_ids = rng.choice(list(latent_dict.keys()), size=N_HUMANS, replace=False)
active_latents = np.stack([latent_dict[w] for w in active_worker_ids])  # (20, 16)
active_risks = np.array([risk_dict[w] for w in active_worker_ids])  # (20,)

# Entity Indices for State Vector
# State = [Mill_Backlogs(3), Crane_States(2*3), Cobot_States(5*3), Human_States(20*(2+16+1))]
# Human State: [x, y, fatigue, latent(16), risk(1)] -> 20 dims per human
STATE_DIM = N_MILLS + N_CRANES * 3 + N_COBOTS * 3 + N_HUMANS * 20
ACTION_DIM = (
    N_HUMANS + N_COBOTS + N_CRANES
)  # Assign each entity a task ID (simplified: 0=Idle, 1=Mill1, 2=Mill2, 3=Mill3, 4=Scrap, 5=Maintenance)
# Actually, Action Space: Discrete per entity is huge.
# SIMPLIFICATION: Agent outputs GLOBAL assignment policy logits -> Sample assignment.
# Let's use MultiDiscrete: One action per entity (Task Assignment).
N_TASKS = 6  # Idle, Mill1, Mill2, Mill3, Scrap, Maintenance
ACTION_SPACE = spaces.MultiDiscrete([N_TASKS] * (N_HUMANS + N_COBOTS + N_CRANES))


# ==============================================================================
# 3. GYMNASIUM ENVIRONMENT
# ==============================================================================
class ShopfloorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, active_latents, active_risks, worker_ids):
        super().__init__()
        self.cfg = cfg
        self.latents = active_latents  # (20, 16)
        self.risks = active_risks  # (20,)
        self.worker_ids = worker_ids
        self.n_humans = N_HUMANS
        self.n_cobots = N_COBOTS
        self.n_cranes = N_CRANES
        self.n_mills = N_MILLS

        self.action_space = ACTION_SPACE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

        self.reset()

    def _get_obs(self):
        # Normalize positions to [0,1]
        obs = []
        obs.extend(self.mill_backlogs / 50.0)  # 3
        for c in self.cranes:
            obs.extend([c["pos"][0] / 200, c["pos"][1] / 100, c["load"]])  # 3 each
        for c in self.cobots:
            obs.extend([c["pos"][0] / 200, c["pos"][1] / 100, c["util"]])  # 3 each
        for i, h in enumerate(self.humans):
            # Human: x, y, fatigue, latent(16), risk(1)
            obs.extend(
                [
                    h["pos"][0] / 200,
                    h["pos"][1] / 100,
                    h["fatigue"],
                    *self.latents[i],  # 16
                    self.risks[i],  # 1
                ]
            )
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Mill Backlogs
        self.mill_backlogs = rng.poisson(lam=5, size=self.n_mills).astype(float)
        # Cranes
        self.cranes = [
            {
                "pos": np.array([rng.uniform(50, 150), rng.uniform(20, 80)]),
                "load": 0.0,
                "target": None,
            }
            for _ in range(self.n_cranes)
        ]
        # Cobots (Fixed stations)
        self.cobots = [
            {"pos": np.array([rng.uniform(50, 150), rng.uniform(30, 70)]), "util": 0.0}
            for _ in range(self.n_cobots)
        ]
        # Humans
        self.humans = [
            {
                "pos": np.array([rng.uniform(10, 190), rng.uniform(10, 90)]),
                "fatigue": rng.uniform(0.1, 0.3),
            }
            for _ in range(self.n_humans)
        ]
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        reward = 0.0

        # --- DECODE ACTION ---
        # action is array of len (20+5+2)=27, values 0-5
        human_actions = action[: self.n_humans]
        cobot_actions = action[self.n_humans : self.n_humans + self.n_cobots]
        crane_actions = action[-self.n_cranes :]

        # --- DYNAMICS ---
        # 1. Mill Processing (Autonomous consumption)
        processed = np.minimum(
            self.mill_backlogs, rng.poisson(lam=2, size=self.n_mills)
        )
        self.mill_backlogs -= processed
        reward += np.sum(processed) * 1.0  # Throughput reward

        # 2. Crane Logic (Move coils Mill -> Scrap)
        for i, c in enumerate(self.cranes):
            act = crane_actions[i]
            if c["load"] == 0 and act in [1, 2, 3]:  # Pick from Mill
                idx = act - 1
                if self.mill_backlogs[idx] > 0:
                    self.mill_backlogs[idx] -= 1
                    c["load"] = 1.0
                    c["target"] = np.array([180.0, 50.0])  # Scrap yard
            elif c["load"] == 1 and act == 4:  # Drop at Scrap
                c["load"] = 0.0
                c["target"] = None
                reward += 0.5  # Scrap cleared
            # Move
            if c["target"] is not None:
                vec = c["target"] - c["pos"]
                dist = np.linalg.norm(vec)
                if dist < 5:
                    c["pos"] = c["target"]
                    c["target"] = None
                else:
                    c["pos"] += (vec / dist) * 10.0  # Speed

        # 3. Cobot Logic (Assist Mill / Process Scrap)
        for i, c in enumerate(self.cobots):
            act = cobot_actions[i]
            if act in [1, 2, 3]:  # Assist Mill
                idx = act - 1
                if self.mill_backlogs[idx] > 0:
                    self.mill_backlogs[idx] -= 1
                    c["util"] = min(1.0, c["util"] + 0.2)
                    reward += 0.8
            elif act == 4:  # Process Scrap
                c["util"] = min(1.0, c["util"] + 0.1)
                reward += 0.3
            else:
                c["util"] *= 0.9

        # 4. Human Logic (Walk, Fatigue, Risk Penalty)
        for i, h in enumerate(self.humans):
            act = human_actions[i]
            risk = self.risks[i]
            fatigue = h["fatigue"]

            # Fatigue Dynamics
            if act != 0:  # Working
                fatigue = min(
                    1.0, fatigue + 0.02 + risk * 0.01
                )  # High risk -> faster fatigue
                h["fatigue"] = fatigue
                # Move towards task zone
                target_zone = self._get_zone_center(act)
                vec = target_zone - h["pos"]
                dist = np.linalg.norm(vec)
                if dist > 2:
                    h["pos"] += (vec / dist) * 1.5  # Walk speed
            else:  # Resting
                fatigue = max(0.0, fatigue - 0.01)
                h["fatigue"] = fatigue

            # Reward Penalties
            reward -= fatigue * 0.5  # Fatigue cost
            reward -= risk * 0.5  # Safety risk cost (high risk worker working)
            if act != 0 and fatigue > 0.8:
                reward -= 2.0  # Safety violation

        # 5. New Arrivals (Stochastic)
        self.mill_backlogs += rng.poisson(lam=1.5, size=self.n_mills)

        # Done condition
        terminated = self.step_count >= self.cfg["max_steps"]
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_zone_center(self, task_id):
        # Map Task ID -> Factory Coordinate
        if task_id == 0:
            return np.array([100, 50])  # Idle Center
        if task_id in [1, 2, 3]:
            return np.array([20 + (task_id - 1) * 50, 30])  # Mill Zones
        if task_id == 4:
            return np.array([180, 50])  # Scrap
        if task_id == 5:
            return np.array([50, 80])  # Maintenance
        return np.array([100, 50])


# ==============================================================================
# 4. DQN AGENT (Multi-Discrete Action Handling via Independent Heads)
# ==============================================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim_per_entity, n_entities):
        super().__init__()
        self.n_entities = n_entities
        self.action_dim = action_dim_per_entity
        self.net = nn.Sequential(
            nn.Linear(state_dim, CFG["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(CFG["hidden_dim"], CFG["hidden_dim"]),
            nn.ReLU(),
        )
        # Independent Q-head for each entity (Parameter sharing optional, but separate heads clearer)
        self.heads = nn.ModuleList(
            [
                nn.Linear(CFG["hidden_dim"], action_dim_per_entity)
                for _ in range(n_entities)
            ]
        )

    def forward(self, x):
        feat = self.net(x)
        # Output: (Batch, N_Entities, N_Actions)
        return torch.stack([h(feat) for h in self.heads], dim=1)


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# 5. TRAINING LOOP
# ==============================================================================
print("[2/5] Initializing Env & Agent...")
env = ShopfloorEnv(CFG, active_latents, active_risks, active_worker_ids)
N_ENTITIES = N_HUMANS + N_COBOTS + N_CRANES

policy_net = QNetwork(STATE_DIM, N_TASKS, N_ENTITIES).to(CFG["device"])
target_net = QNetwork(STATE_DIM, N_TASKS, N_ENTITIES).to(CFG["device"])
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=CFG["lr"])
buffer = ReplayBuffer(CFG["buffer_size"])
eps = CFG["eps_start"]
episode_rewards = []

CKPT = Path(CFG["paths"]["models"]) / "dqn_shopfloor.pth"
if CKPT.exists():
    print("🔄 Resuming from checkpoint...")
    ckpt = torch.load(CKPT, map_location=CFG["device"])
    policy_net.load_state_dict(ckpt)
    target_net.load_state_dict(ckpt)


def select_action(state, eps):
    if random.random() < eps:
        return np.array(
            [rng.integers(N_TASKS) for _ in range(N_ENTITIES)], dtype=np.int64
        )
    else:
        with torch.no_grad():
            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=CFG["device"]
            ).unsqueeze(0)
            q_vals = policy_net(state_t)  # (1, N_Ent, N_Act)
            return q_vals.argmax(dim=2).cpu().numpy().squeeze(0)


def optimize():
    if len(buffer) < CFG["batch_size"]:
        return
    batch = buffer.sample(CFG["batch_size"])

    # Allocate directly to the device to prevent PCIe bus traffic jams
    state_b = torch.as_tensor(
        np.array(batch.state), dtype=torch.float32, device=CFG["device"]
    )
    action_b = torch.as_tensor(
        np.array(batch.action), dtype=torch.int64, device=CFG["device"]
    )  # (B, N_Ent)
    reward_b = torch.as_tensor(
        np.array(batch.reward), dtype=torch.float32, device=CFG["device"]
    ).unsqueeze(1)
    next_state_b = torch.as_tensor(
        np.array(batch.next_state), dtype=torch.float32, device=CFG["device"]
    )
    done_b = torch.as_tensor(
        np.array(batch.done), dtype=torch.float32, device=CFG["device"]
    ).unsqueeze(1)
    # --- CURRENT Q (Must have grad) ---
    q_vals = policy_net(state_b)  # (B, N_Ent, N_Act)
    q_sa = q_vals.gather(2, action_b.unsqueeze(2)).squeeze(2)  # (B, N_Ent)
    q_sa_mean = q_sa.mean(dim=1, keepdim=True)  # (B, 1)  <-- OUTSIDE no_grad

    # --- TARGET Q (No grad) ---
    with torch.no_grad():
        next_q = target_net(next_state_b).max(dim=2)[0]  # (B, N_Ent)
        next_q_mean = next_q.mean(dim=1, keepdim=True)  # (B, 1)
        target = reward_b + CFG["gamma"] * next_q_mean * (1 - done_b)

    loss = nn.MSELoss()(q_sa_mean, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


print(f"[3/5] Training DQN for {CFG['episodes']} episodes...")
for ep in tqdm(range(1, CFG["episodes"] + 1)):
    state, _ = env.reset()
    ep_reward = 0
    losses = []

    for step_idx in range(CFG["max_steps"]):
        action = select_action(state, eps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward

        # Only run the heavy backpropagation every 4 steps (Standard DQN Practice)
        if step_idx % 4 == 0:
            loss = optimize()
            if loss:
                losses.append(loss)

    eps = max(CFG["eps_end"], eps * CFG["eps_decay"])
    episode_rewards.append(ep_reward)

    if ep % CFG["target_update"] == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if ep % 200 == 0:
        avg_r = np.mean(episode_rewards[-100:])
        print(
            f"  Ep {ep:4d} | AvgReward(100): {avg_r:7.2f} | Eps: {eps:.3f} | Buffer: {len(buffer)}"
        )

# Save Model
torch.save(policy_net.state_dict(), Path(CFG["paths"]["models"]) / "dqn_shopfloor.pth")
print("✅ Model Saved.")

# ==============================================================================
# 6. VISUALIZATIONS (Day 4)
# ==============================================================================
print("[4/5] Generating Visualizations...")
plot_dir = Path(CFG["paths"]["plots"])

# --- VIZ 1: Topological Workforce Graph ---
# Nodes: Workers (Size=Risk, Color=Risk), Edges: Shared Skills/Dept
G = nx.Graph()
# Add active workers
for i, wid in enumerate(active_worker_ids):
    risk = active_risks[i]
    G.add_node(
        wid,
        risk=risk,
        type="human",
        pos=(active_latents[i, 0] * 10, active_latents[i, 1] * 10),
    )  # Latent space pos

# --- VIZ 1: Topological Workforce Graph ---
G = nx.Graph()
for i, wid in enumerate(active_worker_ids):
    risk = active_risks[i]
    G.add_node(
        wid,
        risk=risk,
        type="human",
        pos=(active_latents[i, 0] * 10, active_latents[i, 1] * 10),
    )
df_raw = pd.read_csv(Path(CFG["paths"]["raw"]) / "Employee_Feature_Matrix.csv")
for skill in df_raw["primary_skill_cluster"].unique():
    G.add_node(f"Skill:{skill}", type="skill", risk=0)
for _, row in df_raw[df_raw["worker_id"].isin(active_worker_ids)].iterrows():
    G.add_edge(row["worker_id"], f"Skill:{row['primary_skill_cluster']}")

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
node_colors = [
    G.nodes[n]["risk"] if G.nodes[n]["type"] == "human" else 0 for n in G.nodes()
]
node_sizes = [
    50 + G.nodes[n]["risk"] * 500 if G.nodes[n]["type"] == "human" else 100
    for n in G.nodes()
]

# FIX: Capture the collection returned by draw_networkx_nodes
nodes = nx.draw_networkx_nodes(
    G, pos, node_color=node_colors, node_size=node_sizes, cmap="Reds", alpha=0.9
)
nx.draw_networkx_edges(G, pos, alpha=0.2)
labels = {n: n for n in G.nodes() if G.nodes[n].get("risk", 0) > 0.7}
nx.draw_networkx_labels(G, pos, labels, font_size=8)
plt.title("Workforce Topology: Latent Space + Skill Links (Red=High Flight Risk)")

# FIX: Pass 'nodes' collection to colorbar
plt.colorbar(nodes, label="Flight Risk Score", shrink=0.8)
plt.axis("off")
plt.tight_layout()
plt.savefig(plot_dir / "topology_workforce.png", dpi=150)
plt.close()

# --- VIZ 2: Spatial Mesh (Final State of one Episode) ---
# Run one deterministic episode for visualization
state, _ = env.reset()
for _ in range(CFG["max_steps"]):
    action = select_action(state, 0.0)  # Greedy
    state, _, terminated, truncated, _ = env.step(action)
    if terminated:
        break

fig, ax = plt.subplots(figsize=(14, 7))
# Factory Bounds
ax.set_xlim(FACTORY_BOUNDS["x"])
ax.set_ylim(FACTORY_BOUNDS["y"])
ax.set_aspect("equal")
ax.set_title("Spatial Mesh: Human-Cobot-Crane Positions (Final Episode)")

# Mills
for i in range(N_MILLS):
    ax.add_patch(
        plt.Rectangle(
            (10 + i * 60, 10),
            40,
            20,
            color="gray",
            alpha=0.3,
            label=f"Mill {i + 1}" if i == 0 else "",
        )
    )
# Scrap
ax.add_patch(
    plt.Rectangle((170, 40), 20, 20, color="brown", alpha=0.3, label="Scrap Yard")
)

# Entities
# Humans
h_x = [h["pos"][0] for h in env.humans]
h_y = [h["pos"][1] for h in env.humans]
h_c = [env.risks[i] for i in range(N_HUMANS)]
sc = ax.scatter(
    h_x,
    h_y,
    c=h_c,
    s=100,
    cmap="Reds",
    marker="o",
    edgecolors="k",
    label="Humans",
    vmin=0,
    vmax=1,
)
# Cobots
c_x = [c["pos"][0] for c in env.cobots]
c_y = [c["pos"][1] for c in env.cobots]
ax.scatter(c_x, c_y, c="blue", s=150, marker="s", label="Cobots")
# Cranes
cr_x = [c["pos"][0] for c in env.cranes]
cr_y = [c["pos"][1] for c in env.cranes]
ax.scatter(cr_x, cr_y, c="green", s=200, marker="^", label="Cranes")

plt.colorbar(sc, ax=ax, label="Human Flight Risk")
ax.legend()
plt.tight_layout()
plt.savefig(plot_dir / "spatial_mesh.png", dpi=150)
plt.close()

# --- VIZ 3: Training Curve ---
plt.figure(figsize=(8, 4))
plt.plot(pd.Series(episode_rewards).rolling(50).mean())
plt.title("DQN Training: Rolling Avg Reward (Window 50)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_dir / "training_curve.png", dpi=150)
plt.close()

print(f"✅ Plots saved to {plot_dir}")

# ==============================================================================
# 7. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("🎉 BLOCK 2 COMPLETE (Days 3 & 4)")
print("=" * 60)
print(f"RL Model:        {CFG['paths']['models']}/dqn_shopfloor.pth")
print(f"Topology Graph:  {plot_dir}/topology_workforce.png")
print(f"Spatial Mesh:    {plot_dir}/spatial_mesh.png")
print(f"Training Curve:  {plot_dir}/training_curve.png")
print("\nNext: Block 3 (Day 5) -> Streamlit Dashboard Integration.")
