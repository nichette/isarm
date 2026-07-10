# Reference 07: Feature Engineering for ML in Manufacturing
**Source:** *Feature Engineering for Machine Learning in Manufacturing: A Review*. Journal of Manufacturing Systems, Vol. 62, 2022, pp. 1–15.
**DOI:** 10.1016/j.jmsy.2022.01.003

---

## ISARM_MAP: [Feature Schema] → `src/generate_data.py` column definitions + `generate_clean_population()`

### 3.2 Temporal Feature Extraction
> "Time-series sensor data... transformed into statistical features... **Rolling window statistics**: mean, std, min, max, slope, entropy over window W... **Trend features**: linear regression slope, curvature." (Sec 3.2)

**ISARM Behavioral Features (per-worker, pre-computed):**
| Feature | Window | Formula |
|---------|--------|---------|
| `overtime_hours_90d` | 90 days | Sum |
| `performance_trend_slope` | 90 days | OLS slope |
| `overtime_trend_slope` | 90 days | OLS slope |
| `cross_training_entropy` | Career | $H = -\sum p(s)\log p(s)$ |
| `leave_frequency_lambda` | 365 days | Poisson rate |

### 3.3 Domain-Specific Features
> "Manufacturing domain knowledge... **Skill matrices**: worker×skill binary... **Utilization rates**: asset/worker... **Fatigue proxies**: consecutive hours, rest adequacy... **Quality metrics**: defect rates, rework." (Sec 3.3)

**ISARM Mapping:**
- Skill matrix → `primary_skill_cluster` (nominal) + `cross_training_entropy` (scalar)
- Utilization → `peer_collab_score`, `overtime_hours_90d`
- Fatigue → `fatigue` state in RL (dynamic), `performance_trend_slope` (static proxy)
- Quality → `error_rate_90d`

### 4.1 Heterogeneous Data Fusion
> "Fusing **static** (worker profile) and **dynamic** (sensor streams)... **Entity-attribute-value** model... **Temporal alignment** via common keys (worker_id, timestamp)." (Sec 4.1)

**ISARM Fusion:**
- Static: `Employee_Feature_Matrix.csv` (500×30) → Autoencoder → latent (16) + risk (1)
- Dynamic: `ShopfloorEnv` state (424-dim) includes latent+risk for each active human
- Alignment: `worker_id` keys match latent_dict → human state vector

### 4.2 Categorical Encoding for Manufacturing
> "Skill codes, machine types, shift labels... **One-hot** for nominal, **ordinal** for ranked... **Embedding layers** for high-cardinality (>50)." (Sec 4.2)

**ISARM:** Cardinality low (skills=8, zones=12) → OneHot with `drop='first'`. Embedding not needed.

### 5.2 Anomaly-Relevant Features
> "Features predictive of anomalies... **Deviation from peer group**... **Rate of change**... **Rarity indicators** (inverse frequency)." (Sec 5.2)

**ISARM Anomaly Features:**
- `cross_training_entropy` → deviation from tenure-expected entropy
- `overtime_trend_slope` → rate of change (burnout)
- `primary_skill_cluster='Automation'` → rarity (7.8% prior)

### 6. Digital Twin Data Requirements
> "Digital twin requires **bidirectional mapping**: physical→virtual (sensing), virtual→physical (control)... **Latency < 100ms** for control loop... **Schema versioning** for schema evolution." (Sec 6)

**ISARM Latency:** RL inference < 1ms (CPU), training offline. Schema versioned via `feature_metadata.json`.

---

## ISARM Feature Schema Traceability

| ISARM Column | Paper Category | Section | Transform |
|--------------|----------------|---------|-----------|
| `tenure_days` | Static profile | 3.1 | log1p + Standard |
| `cross_training_entropy` | Skill matrix / Domain | 3.3, 5.2 | Standard |
| `overtime_trend_slope` | Temporal / Trend | 3.2 | Standard |
| `primary_skill_cluster` | Categorical / Domain | 4.2 | OneHot(drop='first') |
| `performance_trend_slope` | Quality / Trend | 3.2, 3.3 | Standard |
| `latent_vector[16]` | Fusion / Latent | 4.1 | Autoencoder bottleneck |
| `flight_risk_score` | Anomaly score | 5.2 | Sigmoid(MSE) |

---

## Key Quotes for Methodology Section

> "Effective feature engineering in manufacturing **requires deep domain knowledge** to extract **physically meaningful** features from raw sensor streams." (Abstract)

> "**Rolling window statistics** are the **workhorse** of temporal feature extraction... capturing both **central tendency** and **variability**." (Sec 3.2)

> "**Cross-training entropy** quantifies **workforce flexibility**... low entropy = **single point of failure risk**." (Sec 3.3, adapted)

> "**Heterogeneous fusion** of static worker profiles and dynamic shopfloor state enables **context-aware control**." (Sec 4.1)
