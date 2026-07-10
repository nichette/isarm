# Reference 09: Streamlit — Deployment & Caching
**Source:** Streamlit Documentation (v1.35+).
**URL:** https://docs.streamlit.io/

---

## ISARM_MAP: [Dashboard] → `src/app.py`

### Caching Primitives

#### `@st.cache_resource`
> "Caches **global resources**... ML models, DB connections, clients... **Runs once per session**... Thread-safe... Use for **immutable objects**."

**ISARM Usage:**

python

@st.cache_resource

def load_preprocessor():

return joblib.load("models/preprocessor.joblib")

@st.cache_resource

def load_autoencoder(input_dim, bottleneck=16):

model = DeepAutoencoder(input_dim, bottleneck)

model.load_state_dict(torch.load("models/autoencoder_best.pth", map_location="cpu"))

model.eval()

return model

@st.cache_resource

def load_dqn(state_dim, action_dim, n_entities):

model = QNetwork(state_dim, action_dim, n_entities)

model.load_state_dict(torch.load("models/dqn_shopfloor.pth", map_location="cpu"))

model.eval()

return model

*All three models loaded once, shared across users.*

#### `@st.cache_data`
> "Caches **data transformations**... DataFrames, arrays, computed results... **Serializable**... Use for **mutable data**."

**ISARM Usage:**

python

@st.cache_data

def load_data():

df_risk = pd.read_csv("outputs/flight_risk_scores.csv")

df_risk['latent_vector'] = df_risk['latent_vector'].apply(parse_latent)

df_raw = pd.read_csv("data/raw/Employee_Feature_Matrix.csv")

return df_risk, df_raw, threshold, meta

@st.cache_data

def build_graph(_df_risk, _df_raw):

# NetworkX graph construction

return G

*CSV parsing + graph build cached; underscore args (`_df_risk`) tell Streamlit to hash by identity not value.*

### Plotly Integration
> "`st.plotly_chart(fig, width='stretch')`... renders interactive Plotly figure... `theme=None` preserves figure's own theme."

**ISARM:** `width='stretch'` (replaces deprecated `use_container_width=True`). `theme=None` preserves dark template.

### Components HTML Fallback
> "`streamlit.components.v1.html(html, height=...)`... embeds raw HTML/JS... Use for Plotly `fig.to_html(include_plotlyjs='cdn')` if `st.plotly_chart` fails."

**ISARM:** Commented fallback in Tab 2 for robustness.

### Session State
> "`st.session_state`... persists across reruns... Use for user inputs, wizard steps."

**ISARM:** Not needed (stateless dashboard). Filters re-applied on each render.

### Layout
> "`st.tabs([...])`... `st.columns([...])`... `st.sidebar`... `st.expander(...)`."

**ISARM:** 5 tabs, sidebar for metrics, expanders for model cards.

### Deployment Config
> "`streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true`... `.streamlit/config.toml` for persistent config."

**ISARM Dockerfile:**

dockerfile

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

### Health Check
> "`/_stcore/health`... returns 200 if app healthy... Use in Docker `HEALTHCHECK` and K8s `livenessProbe`."

**ISARM Dockerfile:**

dockerfile

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

---

## Performance Best Practices (Applied)

| Practice | ISARM Implementation |
|----------|---------------------|
| Cache models | `@st.cache_resource` on all 3 `load_*` functions |
| Cache data | `@st.cache_data` on CSV loads + graph build |
| Minimize recompute | Pure functions; no side effects in cached fns |
| Lazy imports | Heavy imports (torch, plotly) at module level (OK for dashboard) |
| Static assets | `outputs/plots/*.png` served via `st.image()` |

---

## Common Pitfalls Avoided

| Pitfall | Fix |
|---------|-----|
| `use_container_width` deprecation | Use `width='stretch'` |
| Mutable default args in cached fn | Use `_arg` prefix to skip hashing |
| Model not in `eval()` mode | `model.eval()` after `load_state_dict` |
| CUDA OOM on shared GPU | `map_location="cpu"` for dashboard |
| Plotly not rendering | Fallback to `components.v1.html` |
