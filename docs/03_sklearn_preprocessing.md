# Reference 03: Scikit-Learn — Preprocessing Data
**Source:** Scikit-Learn Developers. *User Guide: Preprocessing Data*.
**Version:** 1.3+ (2023)
**URL:** https://scikit-learn.org/stable/modules/preprocessing.html

---

## ISARM_MAP: [Feature Pipeline] → `src/generate_data.py::get_preprocessor()`

### 4.3.1 StandardScaler
> "Standardize features by removing the mean and scaling to unit variance... The standard score of a sample x is calculated as: z = (x - u) / s."
> **Warning:** "Naive application... to sparse data will destroy sparsity... not suitable for sparse matrices."

**ISARM Use:** Applied to `TEMPORAL` (after log1p) and `BEHAVIORAL` columns — both dense, Gaussian-ish after transform. Output verified: `mean ≈ 0, std ≈ 1`.

### 4.3.2 MinMaxScaler
> "Transform features by scaling each feature to a given range... default [0, 1]... preserves zero entries in sparse data."

**ISARM Use:** Applied to `ORDINAL` columns (bounded integers 1–5). Preserves ordinal semantics in [0,1] range.

### 4.3.3 RobustScaler
> "Scaling using statistics that are robust to outliers... uses the interquartile range (IQR)... default range 25th–75th percentile."

**ISARM Note:** Considered for `TEMPORAL` (heavy tails) but `log1p + StandardScaler` chosen per Chapter 4.4 of Goodfellow.

### 4.4.1 OneHotEncoder
> "Encode categorical features as a one-hot numeric array... `handle_unknown='ignore'` avoids error on unseen categories... `drop='first'` avoids collinearity (dummy variable trap)."

**ISARM Use:** `NOMINAL` columns (skill_cluster: 8 cats, zone: 12 cats, etc.). `drop='first'` → 7+11+... dims. `handle_unknown='ignore'` critical for Day 3 inference on new hires.

### 4.5.1 KBinsDiscretizer
> "Bin continuous features into intervals... useful for non-linear relationships."

**ISARM Note:** Not used — continuous features kept continuous for neural net.

### 4.6.1 ColumnTransformer
> "Apply different transformations to different columns... combines transformers horizontally... `remainder='drop'` discards unlisted columns."

**ISARM Pipeline:**

python

ColumnTransformer([

('temporal', Pipeline([('log', FunctionTransformer(np.log1p)), ('scale', StandardScaler())]), TEMPORAL),

('ordinal', MinMaxScaler(), ORDINAL),

('nominal', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), NOMINAL),

('behavioral', StandardScaler(), BEHAVIORAL),

], remainder='drop')

### 4.7.1 FunctionTransformer
> "Construct a transformer from an arbitrary callable... `feature_names_out='one-to-one'` preserves names through Pipeline."

**ISARM Use:** `np.log1p` inside temporal pipeline. `feature_names_out='one-to-one'` required for `get_feature_names_out()` to work.

---

## ISARM-Specific Pipeline Decisions

| Column Group | Raw Type | Pipeline | Rationale |
|--------------|----------|----------|-----------|
| `TEMPORAL` | Counts (heavy tail) | `log1p → StandardScaler` | Log normalizes tenure/OT; Standard centers for NN |
| `ORDINAL` | Bounded 1–5 | `MinMaxScaler` | Preserves order, maps to [0,1] |
| `NOMINAL` | Strings | `OneHotEncoder(drop='first')` | No ordinal assumption; avoids dummy trap |
| `BEHAVIORAL` | Stats (~Gaussian) | `StandardScaler` | Already normalized moments |

---

## Common Pitfalls (Avoided in ISARM)

| Pitfall | Scikit-Learn Warning | ISARM Fix |
|---------|---------------------|-----------|
| Fit on full data | "Fit only on training data" | `pre.fit(X_train); pre.transform(X_val)` |
| Scale OneHot output | "Destroys 0/1 geometry" | No scaling after OneHot |
| Drop all categories | "Use `drop='first'` or `drop='if_binary'`" | `drop='first'` |
| Unknown categories at inference | "Raise error by default" | `handle_unknown='ignore'` |

---

## Feature Names Tracking
> "Since 1.0, `ColumnTransformer.get_feature_names_out()` returns names for all transformers... requires each transformer to implement `get_feature_names_out`."

**ISARM:** `FunctionTransformer(..., feature_names_out='one-to-one')` ensures temporal names propagate. Output: 45 feature names used in `feature_metadata.json`.
