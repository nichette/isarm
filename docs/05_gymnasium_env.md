# Reference 05: Gymnasium — Custom Environments
**Source:** Farama Foundation. *Gymnasium Documentation: Creating Custom Environments*.
**Version:** 0.29+ (2024)
**URL:** https://gymnasium.farama.org/tutorials/gymnasium_basics/creating_custom_environments/

---

## ISARM_MAP: [ShopfloorEnv] → `src/train_rl.py::ShopfloorEnv`

### Required Methods
> "Every environment must implement: `__init__`, `reset()`, `step(action)`, `render()` (optional), `close()` (optional)."

**ISARM Implementation:**

python

class ShopfloorEnv(gym.Env):

def __init__(self, cfg, latents, risks, worker_ids): ...

def reset(self, seed=None, options=None): ...

def step(self, action): ...

# render not needed (headless training)

### Observation Space
> "Must be a `gymnasium.spaces.Space` instance... `Box` for continuous, `Discrete`/`MultiDiscrete` for discrete."

**ISARM:**

python

self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32)

Continuous 424-dim vector — matches DQN input.

### Action Space
> "`MultiDiscrete([n1, n2, ...])` for tuple of discrete actions... each dimension independent."

**ISARM:**

python

self.action_space = spaces.MultiDiscrete([N_TASKS] N_ENTITIES)
N_TASKS=6, N_ENTITIES=27 → 27 independent discrete choices

### reset()
> "Returns `(observation, info)`... `info` is dict for auxiliary info... must accept `seed` for reproducibility."

**ISARM:**

python

def reset(self, seed=None, options=None):

super().reset(seed=seed) # seeds self.np_random

# initialize mill_backlogs, cranes, cobots, humans

return self._get_obs(), {}

### step()
> "Returns `(observation, reward, terminated, truncated, info)`... `terminated` = task complete, `truncated` = time limit."

**ISARM:**

python

def step(self, action):

# decode action → entity assignments

# physics: move cranes, update fatigue, process mills

# compute reward

terminated = self.step_count >= self.cfg["max_steps"]

truncated = False

return self._get_obs(), reward, terminated, truncated, {}

### Reward Design
> "Reward should be scalar... dense rewards learn faster... shaped rewards must preserve optimal policy."

**ISARM Reward (Dense):**

python

reward = (throughput 1.0)

- (fatigue 0.5)

- (risk 0.5)

- (safety_violation 2.0)

All components computable at each step — no sparse terminal-only reward.

### Vectorization (Not Used)
> "`gymnasium.vector.VectorEnv` runs multiple envs in parallel... speeds up data collection."

**ISARM Note:** Single env with large replay buffer (50k) sufficient for 2000 episodes. Vectorization would require batched DQN update — future optimization.

### Seeding
> "`reset(seed=...)` seeds the environment's `np_random`... use `super().reset(seed=seed)`."

**ISARM:** `rng = np.random.default_rng(seed)` in `__init__`, `super().reset(seed=seed)` in `reset()`.

---

## ISARM-Specific Extensions

| Extension | Standard Gym | ISARM Implementation |
|-----------|--------------|---------------------|
| Latent Injection | N/A | Constructor args `latents`, `risks` passed to `_get_obs()` |
| Multi-Entity State | Single agent obs | Concatenated 424-dim vector |
| MultiDiscrete Action | Single discrete | 27×6 independent heads |
| Reward Shaping | Scalar only | Decomposed: throughput/fatigue/risk |
| Deterministic Physics | Optional | Fixed `dt=1min`, linear interpolation |

---

## Testing Checklist (from docs)
- [ ] `env = ShopfloorEnv(...); obs, info = env.reset(); assert obs.shape == (424,)`
- [ ] `action = env.action_space.sample(); obs, r, term, trunc, info = env.step(action)`
- [ ] `env.reset(seed=42)` produces same initial state twice
- [ ] `env.action_space.contains(action)` for all generated actions
