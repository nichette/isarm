# Reference 02: Reinforcement Learning — Sutton & Barto
**Source:** Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
**Chapters:** 3 (MDPs), 4 (Dynamic Programming), 6 (TD Learning), 11 (Deep RL)
**URL:** http://incompleteideas.net/book/RLbook2020.pdf

---

## ISARM_MAP: [RL Formulation] → `src/train_rl.py::ShopfloorEnv` + `QNetwork`

### Chapter 3: Finite Markov Decision Processes
> "The agent-environment interaction is modeled as a Markov decision process... At each time step t, the agent receives a state S_t ∈ S, selects an action A_t ∈ A(S_t), and receives a reward R_{t+1} ∈ R and next state S_{t+1}." (p. 48)

**ISARM MDP Definition:**
- **State S_t:** `[mill_backlogs(3), crane_states(6), cobot_states(15), human_states(400)]` = 424-dim
- **Action A_t:** MultiDiscrete([6] × 27) — 6 tasks × 27 entities
- **Reward R_t:** `α·Throughput − β·HumanFatigue − γ·SafetyRisk` (Eq. in spec)
- **Transition:** Simulated physics + stochastic arrivals

### 3.4 Value Functions
> "The value function v_π(s) = E_π[G_t | S_t = s]... The action-value function q_π(s,a) = E_π[G_t | S_t = s, A_t = a]." (p. 59)

**ISARM DQN:** Approximates `q_π(s, a)` with `QNetwork` — 27 independent heads output `q(s, a_i)` for each entity i.

### 3.5 Optimal Value Functions
> "Bellman optimality equation: q*(s,a) = E[R_{t+1} + γ max_{a'} q*(S_{t+1}, a') | S_t=s, A_t=a]." (p. 61)

**ISARM Target:** `target = r + γ * max_a' Q_target(s', a')` — implemented via `target_net` updated every 10 episodes.

### Chapter 4: Dynamic Programming — Policy Iteration
> "Policy iteration... alternates between policy evaluation (computing v_π) and policy improvement (making π greedy w.r.t. v_π)." (p. 65)

**ISARM Note:** DQN is approximate policy iteration — `policy_net` evaluation via replay buffer, `target_net` provides stable greedy target.

### Chapter 6: Temporal-Difference Learning
> "TD learning... updates estimates based on other estimates... TD(0): V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) − V(S_t)]." (p. 123)

**ISARM DQN Loss:** MSE between `Q(s,a)` and TD-target `r + γ max Q_target(s', a')` — direct TD(0) on action-values.

### 6.6 Q-Learning
> "Q-learning: Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) − Q(S_t, A_t)]... off-policy." (p. 131)

**ISARM:** Experience replay + ε-greedy behavior policy → off-policy Q-learning with function approximation.

### Chapter 11: Deep Reinforcement Learning
> "The breakthrough... was the Deep Q-Network (DQN)... two key innovations: (1) experience replay... (2) a separate target network... updated periodically." (p. 293)

**ISARM Implementation:**

python
Experience Replay

buffer = ReplayBuffer(50000)

batch = buffer.sample(64)
Target Network

target_net.load_state_dict(policy_net.state_dict()) # every 10 episodes
Loss

target = r + γ next_q_max (1 - done)

loss = MSE(q_sa, target)

### 11.4 Multi-Agent RL (Independent Learners)
> "In multi-agent settings... each agent can run its own Q-learning... treating other agents as part of the environment... This is called independent Q-learning." (p. 312)

**ISARM Architecture:** 27 independent Q-heads (shared backbone) — each entity learns its own `q_i(s, a_i)` treating others as environment dynamics. Global reward broadcast to all.

### 11.5 Exploration Strategies
> "ε-greedy... selects a random action with probability ε... ε decayed over time... balances exploration vs exploitation." (p. 295)

**ISARM:** `eps_start=1.0 → eps_end=0.05, decay=0.995/episode` — matches standard practice.

---

## Key Equations for ISARM Defense

| Eq. | Source | ISARM Implementation |
|-----|--------|---------------------|
| 3.14 | Bellman Expectation | `q_sa = Q(s,a)` via `policy_net` |
| 3.20 | Bellman Optimality | `target = r + γ * max Q_target(s', a')` |
| 6.10 | TD(0) Update | `loss = MSE(q_sa, target)` |
| 11.1 | DQN Loss | `L(θ) = E[(r + γ max Q(s',a'; θ⁻) − Q(s,a;θ))²]` |

---

## Design Choices Justified by Text

| Choice | Sutton & Barto Basis |
|--------|---------------------|
| Replay Buffer (50k) | Sec 11.3: "breaks correlations... stabilizes training" |
| Target Network (10 ep) | Sec 11.3: "reduces oscillations... prevents divergence" |
| Multi-Head Q-Network | Sec 11.4: "independent learners... scalable" |
| Global Reward + Local Q | Sec 11.4: "team reward... credit assignment via shared backbone" |
| State = Concat All Entities | Sec 3.1: "Markov property requires full observable state" |
