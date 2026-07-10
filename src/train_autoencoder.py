import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import joblib
from tqdm import tqdm

# Report
from sklearn.metrics import classification_report

# --- CONFIG ---
CFG = {
    "seed": 42,
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "wd": 1e-5,
    "bottleneck": 16,
    "threshold_pct": 95,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "paths": {
        "raw": "data/raw",
        "processed": "data/processed",
        "models": "models",
        "outputs": "outputs",
    },
}
for p in CFG["paths"].values():
    Path(p).mkdir(parents=True, exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
print(f"Device: {CFG['device']}")


# --- MODEL ---
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
        return self.decoder(self.encoder(x)), self.encoder(x)


# --- LOAD DATA ---
print("[1/4] Loading Processed Data...")
data = np.load(Path(CFG["paths"]["processed"]) / "split_data.npz", allow_pickle=True)
X_tr, X_val = data["X_train"], data["X_val"]
y_tr, y_val = data["y_train"], data["y_val"]
input_dim = X_tr.shape[1]
print(f"    Input Dim: {input_dim}")

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_tr)),
    batch_size=CFG["batch_size"],
    shuffle=True,
    pin_memory=True,
)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val)), batch_size=CFG["batch_size"], shuffle=False
)

# --- INIT ---
model = DeepAutoencoder(input_dim, CFG["bottleneck"]).to(CFG["device"])
criterion = nn.MSELoss(reduction="none")
opt = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=10, factor=0.5)

# --- TRAIN ---
print("[2/4] Training...")
best = float("inf")
for ep in range(1, CFG["epochs"] + 1):
    model.train()
    tr_losses = []
    for (b,) in train_loader:
        b = b.to(CFG["device"])
        opt.zero_grad()
        recon, _ = model(b)
        loss = criterion(recon, b).mean(dim=1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for (b,) in val_loader:
            b = b.to(CFG["device"])
            recon, _ = model(b)
            val_losses.append(criterion(recon, b).mean(dim=1).mean().item())

    avg_tr, avg_val = np.mean(tr_losses), np.mean(val_losses)
    sched.step(avg_val)
    if avg_val < best:
        best = avg_val
        torch.save(
            model.state_dict(), Path(CFG["paths"]["models"]) / "autoencoder_best.pth"
        )
    if ep % 10 == 0 or ep == 1:
        print(
            f"    Ep {ep:3d} | Tr: {avg_tr:.6f} | Val: {avg_val:.6f} | LR: {opt.param_groups[0]['lr']:.2e}"
        )

# --- INFERENCE ON FULL DATASET ---
print("[3/4] Computing Flight Risk Scores...")
model.load_state_dict(
    torch.load(
        Path(CFG["paths"]["models"]) / "autoencoder_best.pth",
        map_location=CFG["device"],
    )
)
model.eval()

# Load RAW df to get IDs & Ground Truth
df_raw = pd.read_csv(Path(CFG["paths"]["raw"]) / "Employee_Feature_Matrix.csv")
META = ["worker_id", "is_anomaly", "anomaly_type"]
X_full = df_raw.drop(columns=META)

# Load Preprocessor & Transform
pre = joblib.load(Path(CFG["paths"]["models"]) / "preprocessor.joblib")
X_full_s = pre.transform(X_full)
full_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_full_s)), batch_size=256, shuffle=False
)

all_err, all_z = [], []
with torch.no_grad():
    for (b,) in full_loader:
        b = b.to(CFG["device"])
        recon, z = model(b)
        all_err.append(criterion(recon, b).mean(dim=1).cpu().numpy())
        all_z.append(z.cpu().numpy())

    # --- INFERENCE ON FULL DATASET ---
    print("[3/4] Computing Flight Risk Scores...")
    model.load_state_dict(
        torch.load(
            Path(CFG["paths"]["models"]) / "autoencoder_best.pth",
            map_location=CFG["device"],
        )
    )
    model.eval()

    # Load RAW df to get IDs & Ground Truth
    df_raw = pd.read_csv(Path(CFG["paths"]["raw"]) / "Employee_Feature_Matrix.csv")
    META = ["worker_id", "is_anomaly", "anomaly_type"]
    X_full = df_raw.drop(columns=META)

    # Load Preprocessor & Transform
    pre = joblib.load(Path(CFG["paths"]["models"]) / "preprocessor.joblib")
    X_full_s = pre.transform(X_full)
    full_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_full_s)), batch_size=256, shuffle=False
    )

    all_err, all_z = [], []
    with torch.no_grad():
        for (b,) in full_loader:
            b = b.to(CFG["device"])
            recon, z = model(b)
            all_err.append(criterion(recon, b).mean(dim=1).cpu().numpy())
            all_z.append(z.cpu().numpy())

    all_err = np.concatenate(all_err)
    all_z = np.vstack(all_z)

    # --- THRESHOLD ON TRAINING SET (Correct Way) ---
    # Pass X_tr (already loaded, correct 400 rows) through model
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr)), batch_size=256, shuffle=False
    )
    train_errs = []
    with torch.no_grad():
        for (b,) in train_loader:
            b = b.to(CFG["device"])
            recon, _ = model(b)
            train_errs.append(criterion(recon, b).mean(dim=1).cpu().numpy())
    train_errs = np.concatenate(train_errs)

    # Only normal (y_tr == 0) training errors for threshold
    train_normal_err = train_errs[y_tr == 0]
    threshold = np.percentile(train_normal_err, CFG["threshold_pct"])
    k = 10.0 / (threshold + 1e-8)
    risk = 1 / (1 + np.exp(-k * (all_err - threshold)))


# --- SAVE OUTPUTS ---
print("[4/4] Saving Outputs...")
df_out = df_raw[["worker_id"]].copy()
df_out["reconstruction_mse"] = all_err
df_out["flight_risk_score"] = risk
df_out["risk_tier"] = pd.cut(
    risk, bins=[-0.01, 0.3, 0.7, 1.01], labels=["Low", "Medium", "Critical"]
)
df_out["anomaly_ground_truth"] = df_raw["anomaly_type"].map(
    {0: "Normal", 1: "Silent_Quitter", 2: "SPOF", 3: "Burnout"}
)
df_out["latent_vector"] = list(all_z)

out_path = Path(CFG["paths"]["outputs"]) / "flight_risk_scores.csv"
df_out.to_csv(out_path, index=False)
np.save(Path(CFG["paths"]["outputs"]) / "latent_vectors.npy", all_z)
with open(Path(CFG["paths"]["models"]) / "threshold.json", "w") as f:
    json.dump(
        {"threshold_mse": float(threshold), "percentile": CFG["threshold_pct"]}, f
    )


# Report
# --- VALIDATION ERRORS (Aligned with y_val) ---
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val)), batch_size=256, shuffle=False
)
val_errs = []
with torch.no_grad():
    for (b,) in val_loader:
        b = b.to(CFG["device"])
        recon, _ = model(b)
        val_errs.append(criterion(recon, b).mean(dim=1).cpu().numpy())
val_err = np.concatenate(val_errs)

# Report (Binarize y_val: 0 vs 1,2,3)
y_val_bin = (y_val != 0).astype(int)
preds = (val_err > threshold).astype(int)

print("\n=== VALIDATION REPORT (Binary: Normal vs Anomaly) ===")
print(
    classification_report(
        y_val_bin, preds, target_names=["Normal", "Anomaly"], zero_division=0
    )
)

print(f"\n✅ DAY 2 DONE. Scores: {out_path} | Threshold: {threshold:.6f}")
