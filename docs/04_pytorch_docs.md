# Reference 04: PyTorch Documentation — Core Modules
**Source:** PyTorch Foundation. *PyTorch 2.3 Documentation*.
**URL:** https://pytorch.org/docs/stable/index.html

---

## ISARM_MAP: [Model Implementation] → `src/train_autoencoder.py` + `src/train_rl.py`

### torch.nn.Module
> "Base class for all neural network modules... Submodules assigned as attributes are registered... Parameters are registered via `nn.Parameter`."

**ISARM Classes:**

python

class DeepAutoencoder(nn.Module):

def __init__(self, input_dim, bottleneck_dim):

super().__init__()

self.encoder = nn.Sequential(...) # registered

self.decoder = nn.Sequential(...)

### torch.nn.Linear
> "Applies a linear transformation: y = xA^T + b... Input: (*, in_features), Output: (*, out_features)."

**ISARM:** All dense layers — encoder 45→64→32→16, decoder 16→32→64→45.

### torch.nn.BatchNorm1d
> "Applies Batch Normalization over a 2D or 3D input... y = (x - E[x]) / √(Var[x] + ε) * γ + β... Uses running estimates during eval."

**ISARM:** After every Linear+ReLU except bottleneck. Critical for stable gradients in deep autoencoder.

### torch.nn.Dropout
> "Randomly zeroes elements with probability p... during training only... `eval()` disables dropout."

**ISARM:** `Dropout(0.2)` in encoder, `Dropout(0.1)` in decoder. Prevents memorization (identity mapping).

### torch.nn.ReLU
> "Rectified Linear Unit: relu(x) = max(0, x)... non-saturating gradient."

**ISARM:** All hidden layers. Bottleneck linear (no activation) per Goodfellow Ch.14.

### torch.nn.MSELoss
> "Measures mean squared error... `reduction='none'` returns loss per sample... `reduction='mean'` averages over batch."

**ISARM Autoencoder:**

python

criterion = nn.MSELoss(reduction='none') # (batch, features)

loss_per_sample = criterion(recon, batch).mean(dim=1) # (batch,)

loss = loss_per_sample.mean() # scalar for backprop

**ISARM DQN:** Same pattern — per-entity Q-values gathered, then averaged.

### torch.optim.AdamW
> "AdamW... decouples weight decay from gradient update... `weight_decay` applies L2 to parameters before adaptive step."

**ISARM:** `optim.AdamW(lr=1e-3, weight_decay=1e-5)` — better generalization than Adam+L2.

### torch.nn.utils.clip_grad_norm_
> "Clips gradient norm of iterable parameters... prevents exploding gradients."

**ISARM:** `clip_grad_norm_(model.parameters(), 1.0)` every step — standard for RL stability.

### torch.utils.data.DataLoader
> "Data loader... `pin_memory=True` speeds up CPU→GPU transfer... `shuffle=True` for training."

**ISARM:**

python

DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True, pin_memory=True)

### torch.save / torch.load
> "`state_dict` contains parameters... `load_state_dict` loads into module... `map_location` for device mapping."

**ISARM:**

python

torch.save(model.state_dict(), "autoencoder_best.pth")

model.load_state_dict(torch.load(..., map_location=device))

### torch.no_grad()
> "Context manager that disables gradient calculation... useful for inference."

**ISARM:** All evaluation loops wrapped in `with torch.no_grad():`.

---

## Device Management

python

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

batch = batch.to(device)

**ISARM:** Consistent in both training scripts. CPU fallback for dashboard.

---

## Key Patterns for Reproducibility

python

torch.manual_seed(42)

np.random.seed(42)

random.seed(42)

torch.backends.cudnn.deterministic = True # if needed

**ISARM:** Seeds set in both `train_autoencoder.py` and `train_rl.py`.

---

