# Reference 08: NetworkX & Graph Visualization Theory
**Sources:**
- NetworkX Documentation: *Drawing* (v3.2+). https://networkx.org/documentation/stable/reference/drawing.html
- Fruchterman, T. & Reingold, E. (1991). *Graph Drawing by Force-Directed Placement*. Software: Practice and Experience.
- Plotly Documentation: *Graph Objects* — Scatter, Layout.

---

## ISARM_MAP: [Topology Graph] → `src/train_rl.py` Viz 1 + `src/app.py` Tab 2

### NetworkX: spring_layout (Fruchterman-Reingold)
> "`spring_layout(G, k=None, iterations=50)`... positions nodes using force-directed algorithm... `k` = optimal distance between nodes... larger `k` spreads graph."

**ISARM:** `k=0.8, iterations=50, seed=42` — deterministic, readable layout for 20 humans + 8 skills = 28 nodes.

### Node Attributes
> "Node attributes accessed via `G.nodes[n]['attr']`... used for visual mapping: size, color, label."

**ISARM Mapping:**

python
Human nodes

risk = G.nodes[n]['risk'] # 0–1 flight risk score

node_size = 10 + risk 40

node_color = risk # mapped to 'Reds' colorscale
Skill nodes

risk = 0, size = 15, color = 0 (gray)

### Bipartite Graph Structure
> "Workforce-skill graph is naturally **bipartite**: Workers — Skills... no worker-worker or skill-skill edges."

**ISARM Construction:**

python

for wid in active_ids:

G.add_node(wid, type='human', risk=risk_map[wid])

for skill in skills:

G.add_node(f"Skill:{skill}", type='skill', risk=0)

for row in df_raw:

G.add_edge(row['worker_id'], f"Skill:{row['primary_skill_cluster']}")

### Edge Weights (Future)
> "Edge weight = shared projects, co-location frequency... `width` in draw_networkx_edges."

**ISARM:** Currently unweighted (single skill per worker). Extensible to multi-skill via weight = proficiency.

---

## Plotly Interactive Rendering

### Scatter Trace for Nodes
> "`go.Scatter(mode='markers+text')`... `marker=dict(size, color, colorscale, showscale, colorbar)`... `text` for hover/label."

**ISARM (Tab 2 Dashboard):**

python

fig.add_trace(go.Scatter(

x=node_x, y=node_y, mode='markers+text',

text=node_text, textposition="top center",

marker=dict(

size=node_size, color=node_color,

colorscale='Reds', showscale=True,

colorbar=dict(title='Flight Risk'), cmin=0, cmax=1

),

hoverinfo='text'

))

### Edge Traces
> "Edges as separate `go.Scatter(mode='lines')` with `None` separators... `hoverinfo='none'`."

**ISARM:**

python

edge_x, edge_y = [], []

for u, v in G.edges():

x0, y0 = pos[u]; x1, y1 = pos[v]

edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#444')))

### Dark Theme (Plotly)
> "`template='plotly_dark'` + `paper_bgcolor='black'` + `plot_bgcolor='black'` + `font=dict(color='white')`."

**ISARM:** Applied in both `train_rl.py` (static PNG) and `app.py` (interactive).

---

## Computational Complexity

| Operation | Complexity | ISARM Scale |
|-----------|------------|-------------|
| `spring_layout` | O(|V|²) per iter | 28 nodes × 50 iter = trivial |
| Node/edge iteration | O(|V|+|E|) | 28 + 20 = trivial |
| Plotly rendering | O(|V|) DOM nodes | 28 traces = instant |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Bipartite (Worker–Skill) | Mirrors organizational reality; no spurious edges |
| Risk → Size/Color | Pre-attentive encoding: pop-out for high risk |
| Force-directed layout | Reveals clusters (skill communities) naturally |
| Interactive (Plotly) | Ops can hover for worker_id, risk value |
| Dark theme | Control-room standard; reduces eye strain |

