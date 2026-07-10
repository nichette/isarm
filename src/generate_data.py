import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- CONFIG ---
CFG = {
    "seed": 42,
    "n_workers": 500,
    "anomaly_ratio": 0.06,
    "test_size": 0.2,
    "paths": {"raw": "data/raw", "processed": "data/processed", "models": "models"},
}
for p in CFG["paths"].values():
    Path(p).mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(CFG["seed"])

# --- SCHEMA ---
TEMPORAL = [
    "tenure_days",
    "days_since_promotion",
    "overtime_hours_90d",
    "total_leaves_365d",
    "avg_leave_duration_days",
]
ORDINAL = [
    "certification_level",
    "role_seniority_band",
    "shift_preference_idx",
    "safety_training_level",
    "mentorship_tier",
]
NOMINAL = [
    "primary_skill_cluster",
    "assignment_zone",
    "employment_type",
    "department_unit",
    "education_band",
]
BEHAVIORAL = [
    "leave_frequency_lambda",
    "leave_duration_variance",
    "cross_training_entropy",
    "performance_trend_slope",
    "overtime_trend_slope",
    "error_rate_90d",
    "peer_collab_score",
]
META = ["worker_id", "is_anomaly", "anomaly_type"]
ALL_COLS = TEMPORAL + ORDINAL + NOMINAL + BEHAVIORAL + META


def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "temporal",
                Pipeline(
                    [
                        (
                            "log",
                            FunctionTransformer(
                                np.log1p, validate=False, feature_names_out="one-to-one"
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                TEMPORAL,
            ),
            ("ordinal", MinMaxScaler(), ORDINAL),
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="first"
                ),
                NOMINAL,
            ),
            ("behavioral", StandardScaler(), BEHAVIORAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def generate_clean(n):
    df = pd.DataFrame(index=range(n))
    df["worker_id"] = [f"W_{i:04d}" for i in range(n)]
    df["is_anomaly"] = 0
    df["anomaly_type"] = 0
    maturity = rng.beta(2, 5, n)
    intensity = rng.gamma(2, 1, n)
    engagement = rng.beta(5, 2, n)
    df["tenure_days"] = (
        (maturity * 4000 + rng.normal(0, 200, n)).clip(30, 5000).astype(int)
    )
    df["days_since_promotion"] = (
        (df["tenure_days"] * rng.uniform(0.05, 0.8, n) + rng.normal(0, 50, n))
        .clip(0)
        .astype(int)
    )
    df["overtime_hours_90d"] = (
        (intensity * 40 + rng.normal(0, 10, n)).clip(0, 250).astype(int)
    )
    df["total_leaves_365d"] = rng.poisson(lam=15 - engagement * 10, size=n).clip(0, 50)
    df["avg_leave_duration_days"] = rng.gamma(2, 2, n).clip(0.5, 30).round(1)
    df["certification_level"] = (
        (maturity * 4 + rng.normal(0, 0.5, n)).clip(1, 5).astype(int)
    )
    df["role_seniority_band"] = (
        (maturity * 3 + rng.normal(0, 0.5, n)).clip(1, 4).astype(int)
    )
    df["shift_preference_idx"] = rng.integers(1, 4, n)
    df["safety_training_level"] = rng.integers(1, 4, n)
    df["mentorship_tier"] = rng.integers(1, 4, n)
    skills = [
        "Rolling",
        "Casting",
        "Finishing",
        "Maintenance",
        "Quality",
        "Logistics",
        "Automation",
        "Safety",
    ]
    sp = [0.20, 0.15, 0.15, 0.15, 0.10, 0.10, 0.08, 0.07]
    df["primary_skill_cluster"] = rng.choice(skills, n, p=sp)
    df["assignment_zone"] = rng.choice([f"Zone_{z}" for z in range(1, 13)], n)
    df["employment_type"] = rng.choice(
        ["FullTime", "Contractor", "Apprentice"], n, p=[0.85, 0.10, 0.05]
    )
    df["department_unit"] = rng.choice(
        ["Mill_Ops", "Maintenance", "Quality", "Logistics", "Admin"], n
    )
    df["education_band"] = rng.choice(
        ["Diploma", "BTech", "MTech", "ITI", "PhD"], n, p=[0.2, 0.3, 0.15, 0.3, 0.05]
    )
    df["leave_frequency_lambda"] = (15 - engagement * 12 + rng.normal(0, 1, n)).clip(
        0.1, 40
    )
    df["leave_duration_variance"] = rng.gamma(1.5, 1.5, n).clip(0.1, 20)
    df["cross_training_entropy"] = (maturity * 1.9 + rng.normal(0, 0.15, n)).clip(
        0.01, 2.0
    )
    df["performance_trend_slope"] = (
        engagement * 0.05 - 0.02 + rng.normal(0, 0.03, n)
    ).clip(-0.2, 0.2)
    df["overtime_trend_slope"] = (intensity * 0.01 + rng.normal(0, 0.01, n)).clip(
        -0.1, 0.1
    )
    df["error_rate_90d"] = (intensity * 0.005 + rng.gamma(1, 0.005, n)).clip(0, 0.15)
    df["peer_collab_score"] = engagement * rng.beta(5, 2, n)
    return df[ALL_COLS]


def inject_anomalies(df, cfg):
    n_anom = int(len(df) * cfg["anomaly_ratio"])
    idx = rng.choice(df.index, size=n_anom, replace=False)
    df = df.copy()

    # Helper to fill a column for specific indices
    def set_col(indices, col, values):
        df.loc[indices, col] = values

    # --- TYPE 1: SILENT QUITTER (10 rows) ---
    i1 = idx[: n_anom // 3]
    if len(i1):
        n1 = len(i1)
        set_col(i1, "tenure_days", rng.integers(2500, 4500, n1))
        set_col(i1, "cross_training_entropy", 0.001)
        set_col(i1, "primary_skill_cluster", "Rolling")
        set_col(i1, "total_leaves_365d", rng.integers(30, 60, n1))
        set_col(i1, "leave_frequency_lambda", rng.uniform(25, 40, n1))
        set_col(i1, "performance_trend_slope", rng.uniform(-0.15, -0.05, n1))
        set_col(i1, "overtime_hours_90d", rng.integers(0, 20, n1))
        set_col(i1, "peer_collab_score", rng.uniform(0, 0.15, n1))
        set_col(i1, "is_anomaly", 1)
        set_col(i1, "anomaly_type", 1)

    # --- TYPE 2: SPOF (10 rows) ---
    i2 = idx[n_anom // 3 : 2 * n_anom // 3]
    if len(i2):
        n2 = len(i2)
        set_col(i2, "primary_skill_cluster", "Automation")
        set_col(i2, "tenure_days", rng.integers(100, 500, n2))
        set_col(i2, "cross_training_entropy", 0.001)
        set_col(i2, "overtime_hours_90d", rng.integers(130, 200, n2))
        set_col(i2, "overtime_trend_slope", rng.uniform(0.05, 0.12, n2))
        set_col(i2, "certification_level", 5)
        set_col(i2, "error_rate_90d", rng.uniform(0.03, 0.08, n2))
        set_col(i2, "is_anomaly", 1)
        set_col(i2, "anomaly_type", 2)

    # --- TYPE 3: BURNOUT (10 rows) ---
    i3 = idx[2 * n_anom // 3 :]
    if len(i3):
        n3 = len(i3)
        set_col(i3, "overtime_hours_90d", rng.integers(160, 220, n3))
        set_col(i3, "overtime_trend_slope", rng.uniform(0.08, 0.18, n3))
        set_col(i3, "performance_trend_slope", rng.uniform(-0.25, -0.1, n3))
        set_col(i3, "total_leaves_365d", rng.integers(0, 2, n3))
        set_col(i3, "leave_frequency_lambda", 0.01)
        set_col(i3, "error_rate_90d", rng.uniform(0.06, 0.15, n3))
        set_col(i3, "peer_collab_score", rng.uniform(0, 0.1, n3))
        set_col(i3, "is_anomaly", 1)
        set_col(i3, "anomaly_type", 3)

    return df


if __name__ == "__main__":
    print("[1/3] Generating Raw Data...")
    df = generate_clean(CFG["n_workers"])
    df = inject_anomalies(df, CFG)
    raw_path = Path(CFG["paths"]["raw"]) / "Employee_Feature_Matrix.csv"
    df.to_csv(raw_path, index=False)
    print(f"    Saved: {raw_path} ({df.shape})")
    print(f"    Anomaly Counts:\n{df['anomaly_type'].value_counts().sort_index()}")

    print("[2/3] Fitting Preprocessor & Splitting...")
    X = df.drop(columns=META)
    y = df["anomaly_type"].values
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=CFG["test_size"], random_state=CFG["seed"], stratify=y
    )
    pre = get_preprocessor()
    X_tr_s = pre.fit_transform(X_tr)
    X_val_s = pre.transform(X_val)
    print(f"    Train Shape: {X_tr_s.shape} | Val Shape: {X_val_s.shape}")
    print(
        f"    Mean/Std Check (first 5): {X_tr_s.mean(axis=0)[:5].round(3)} / {X_tr_s.std(axis=0)[:5].round(3)}"
    )

    print("[3/3] Saving Artifacts...")
    joblib.dump(pre, Path(CFG["paths"]["models"]) / "preprocessor.joblib")
    np.savez(
        Path(CFG["paths"]["processed"]) / "split_data.npz",
        X_train=X_tr_s,
        X_val=X_val_s,
        y_train=y_tr,
        y_val=y_val,
        feature_names=pre.get_feature_names_out(),
    )
    with open(Path(CFG["paths"]["models"]) / "feature_metadata.json", "w") as f:
        json.dump(
            {
                "raw_cols": ALL_COLS,
                "processed_dim": X_tr_s.shape[1],
                "anomaly_map": {
                    0: "Normal",
                    1: "Silent_Quitter",
                    2: "SPOF",
                    3: "Burnout",
                },
            },
            f,
            indent=2,
        )
    print("✅ DAY 1 DONE. Artifacts in /models & /data/processed")
