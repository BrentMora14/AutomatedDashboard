import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import json
import re
import io
import math
from PIL import Image
from google import genai
from google.genai import types as genai_types

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSV Auto-Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f45; }
    .block-container { padding-top: 1.5rem; }
    h1 { font-size: 1.6rem !important; font-weight: 600 !important; letter-spacing: -0.5px; }
    .insight-box {
        background: linear-gradient(135deg, #1a1f35 0%, #161b27 100%);
        border: 1px solid #2a4f8a;
        border-left: 3px solid #4e8af4;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.75rem 0 1.25rem 0;
        font-size: 0.9rem;
        color: #c8d0e8;
        line-height: 1.6;
    }
    .stPlotlyChart { border-radius: 10px; overflow: hidden; }
    .upload-prompt { text-align:center; color:#6b7280; padding: 2rem 0; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Free key from https://aistudio.google.com — no credit card needed",
    )
    st.caption("🔑 [Get a free key →](https://aistudio.google.com/app/apikey)")
    st.divider()
    chart_theme = st.selectbox("Color theme", ["Dark ocean", "Midnight blue", "Forest green", "Warm ember"])
    top_n = st.slider("Top-N categories per chart", 5, 30, 15, step=5,
        help="Max unique category values shown per chart")
    scatter_limit = st.slider("Scatter point limit", 100, 2000, 500, step=100,
        help="Max rows plotted on scatter charts (for performance)")
    st.divider()
    st.markdown("**About**")
    st.caption("AI picks chart types & columns. Python computes all data from your full CSV.")

THEMES = {
    "Dark ocean":    {"bg": "#0f1117", "paper": "#161b27", "grid": "#1e2535", "text": "#c8d0e8"},
    "Midnight blue": {"bg": "#060c1e", "paper": "#0d1530", "grid": "#1a2545", "text": "#b8c8f0"},
    "Forest green":  {"bg": "#0b1209", "paper": "#121f10", "grid": "#1a2f18", "text": "#c0d8b8"},
    "Warm ember":    {"bg": "#130c05", "paper": "#1f1208", "grid": "#2e1c0a", "text": "#e8d0b0"},
}
theme = THEMES[chart_theme]

def get_palette(theme_name):
    palettes = {
        "Dark ocean":    ["#4e8af4","#34c7a0","#f4844e","#c97cf4","#f4c94e","#f4506e","#4ec9f4"],
        "Midnight blue": ["#6b9ff5","#5ec8e8","#c589f5","#f58c6b","#f5d06b","#5ef5b4","#f56b8c"],
        "Forest green":  ["#5ec47a","#a0d45e","#5eb8a0","#d4a05e","#5e8ad4","#d45e7a","#c4d45e"],
        "Warm ember":    ["#f4844e","#f4c94e","#f4506e","#f4a04e","#c4784e","#f4e44e","#e87850"],
    }
    return palettes.get(theme_name, palettes["Dark ocean"])

# ── Column type inference ──────────────────────────────────────────────────────
def infer_column_types(df):
    col_info = {}
    for col in df.columns:
        s = df[col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            col_info[col] = "numeric"
        else:
            try:
                pd.to_datetime(s.head(20))
                col_info[col] = "datetime"
            except Exception:
                col_info[col] = f"categorical ({s.nunique()} unique)"
    return col_info

def compute_summary_stats(df):
    stats = {}
    for col in df.select_dtypes(include="number").columns:
        stats[col] = {
            "min":   float(df[col].min()),
            "max":   float(df[col].max()),
            "mean":  round(float(df[col].mean()), 4),
            "sum":   float(df[col].sum()),
            "count": int(df[col].count()),
        }
    return stats

# ── Gemini API ─────────────────────────────────────────────────────────────────
def call_gemini(api_key, prompt):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    return json.loads(text)

# ── Prompt: AI decides WHAT to show, not the actual data ──────────────────────
def build_prompt(df, col_info, stats, filename):
    cat_samples = {}
    for col, ctype in col_info.items():
        if "categorical" in ctype:
            top = df[col].dropna().value_counts().head(5)
            cat_samples[col] = top.index.tolist()

    return f"""You are an expert data analyst. Analyze this CSV schema and return ONLY valid JSON.
DO NOT return data arrays. Only return column names and aggregation instructions.
Python will compute the actual data from the full dataset.

Filename: {filename}
Total rows: {len(df)}
Column types: {json.dumps(col_info)}
Numeric stats (from all rows): {json.dumps(stats, indent=2)}
Sample category values: {json.dumps(cat_samples)}

Return this exact JSON structure:
{{
  "title": "3-6 word dashboard title",
  "insight": "3-4 sentence executive summary of what the data shows — key patterns, anomalies, trends",
  "metrics": [
    {{
      "label": "display label",
      "col": "exact column name from the dataset",
      "agg": "sum|mean|count|min|max|nunique",
      "format": "currency|number|integer|percent"
    }}
  ],
  "charts": [
    {{
      "id": "unique_id",
      "type": "bar|horizontal_bar|line|area|pie|donut|scatter|histogram|box",
      "title": "chart title",
      "subtitle": "one sentence explaining what this shows",
      "x_col": "exact column name for x-axis (or null)",
      "y_col": "exact column name for y-axis (or null)",
      "col":   "exact column name (for histogram only)",
      "agg":   "sum|mean|count|min|max",
      "sort":  "desc|asc|none",
      "limit": 15,
      "time_group": "day|month|year|null",
      "x_label": "human-readable x axis label or null",
      "y_label": "human-readable y axis label or null"
    }}
  ]
}}

Chart type rules — use ONLY column names that exist in the dataset:
- bar/horizontal_bar: x_col=categorical, y_col=numeric, agg=aggregation to apply
- line/area: x_col=datetime or ordered categorical, y_col=numeric, agg=aggregation, time_group if datetime
- pie/donut: x_col=categorical (<=8 unique values), y_col=numeric, agg=sum
- scatter: x_col=numeric, y_col=numeric (no agg needed)
- histogram: col=numeric column to distribute (no x_col/y_col needed)
- box: y_col=numeric, x_col=categorical grouping (optional)

Metric rules:
- 4-6 KPIs. Use real column names. agg must be one of: sum, mean, count, min, max, nunique
- format: currency if monetary, percent if ratio, integer if count/id, number otherwise

Return 4-7 charts. Return ONLY the JSON object, no other text."""

# ── Python data computation (uses FULL DataFrame) ──────────────────────────────
AGG_MAP = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}

def _safe_float(v):
    try:
        return round(float(v), 6)
    except Exception:
        return 0.0

def compute_metric(spec, df):
    col   = spec.get("col")
    agg   = spec.get("agg", "sum")
    fmt   = spec.get("format", "number")
    label = spec.get("label", col or "")

    if not col or col not in df.columns:
        return label, "N/A"

    s = df[col].dropna()
    try:
        if   agg == "sum":     val = float(s.sum())
        elif agg == "mean":    val = float(s.mean())
        elif agg == "count":   val = float(len(s))
        elif agg == "min":     val = float(s.min())
        elif agg == "max":     val = float(s.max())
        elif agg == "nunique": val = float(s.nunique())
        else:                  val = float(s.sum())
    except Exception:
        return label, "N/A"

    if fmt == "currency":
        if   abs(val) >= 1_000_000_000: formatted = f"${val/1_000_000_000:.2f}B"
        elif abs(val) >= 1_000_000:     formatted = f"${val/1_000_000:.2f}M"
        elif abs(val) >= 1_000:         formatted = f"${val/1_000:.1f}K"
        else:                           formatted = f"${val:,.2f}"
    elif fmt == "percent":
        formatted = f"{val:.1f}%"
    elif fmt == "integer" or agg in ("count", "nunique"):
        formatted = f"{int(val):,}"
    else:
        if   abs(val) >= 1_000_000_000: formatted = f"{val/1_000_000_000:.2f}B"
        elif abs(val) >= 1_000_000:     formatted = f"{val/1_000_000:.2f}M"
        elif abs(val) >= 1_000:         formatted = f"{val/1_000:.1f}K"
        else:                           formatted = f"{val:,.2f}"

    return label, formatted


def compute_chart_data(spec, df, top_n=15, scatter_limit=500):
    ctype  = spec.get("type", "bar")
    x_col  = spec.get("x_col")
    y_col  = spec.get("y_col")
    col    = spec.get("col")
    agg    = spec.get("agg", "sum")
    sort   = spec.get("sort", "desc")
    limit  = int(spec.get("limit") or top_n)
    tgroup = spec.get("time_group")
    agg_fn = AGG_MAP.get(agg, "sum")

    def col_ok(c):
        return c and c in df.columns

    # bar / horizontal_bar / pie / donut
    if ctype in ("bar", "horizontal_bar", "pie", "donut"):
        if not col_ok(x_col) or not col_ok(y_col):
            return None
        grouped = df.groupby(x_col)[y_col].agg(agg_fn).dropna()
        if sort == "desc": grouped = grouped.sort_values(ascending=False)
        elif sort == "asc": grouped = grouped.sort_values(ascending=True)
        grouped = grouped.head(limit)
        return {
            "labels":   [str(k) for k in grouped.index.tolist()],
            "datasets": [{"name": y_col, "data": [_safe_float(v) for v in grouped.values]}],
        }

    # line / area
    elif ctype in ("line", "area"):
        if not col_ok(x_col) or not col_ok(y_col):
            return None
        tmp = df[[x_col, y_col]].dropna().copy()
        is_dt = False
        try:
            tmp[x_col] = pd.to_datetime(tmp[x_col])
            is_dt = True
        except Exception:
            pass

        if is_dt:
            tmp = tmp.sort_values(x_col)
            if   tgroup == "year":  tmp["_grp"] = tmp[x_col].dt.to_period("Y").astype(str)
            elif tgroup == "month": tmp["_grp"] = tmp[x_col].dt.to_period("M").astype(str)
            elif tgroup == "day":   tmp["_grp"] = tmp[x_col].dt.date.astype(str)
            else:                   tmp["_grp"] = tmp[x_col].astype(str)
        else:
            tmp["_grp"] = tmp[x_col].astype(str)

        grouped = tmp.groupby("_grp", sort=False)[y_col].agg(agg_fn).dropna()
        return {
            "labels":   grouped.index.tolist(),
            "datasets": [{"name": y_col, "data": [_safe_float(v) for v in grouped.values]}],
        }

    # scatter
    elif ctype == "scatter":
        if not col_ok(x_col) or not col_ok(y_col):
            return None
        tmp = df[[x_col, y_col]].dropna().head(scatter_limit)
        return {
            "labels":   [_safe_float(v) for v in tmp[x_col].tolist()],
            "datasets": [{"name": f"{x_col} vs {y_col}",
                          "data":  [_safe_float(v) for v in tmp[y_col].tolist()]}],
        }

    # histogram
    elif ctype == "histogram":
        target = col or y_col
        if not col_ok(target):
            return None
        data = df[target].dropna().tolist()
        return {
            "labels":   [],
            "datasets": [{"name": target, "data": [_safe_float(v) for v in data]}],
        }

    # box
    elif ctype == "box":
        if not col_ok(y_col):
            return None
        if col_ok(x_col):
            top_cats = df[x_col].dropna().value_counts().head(limit).index.tolist()
            datasets = [{"name": str(cat),
                         "data": [_safe_float(v) for v in df[df[x_col] == cat][y_col].dropna().tolist()]}
                        for cat in top_cats]
            return {"labels": [], "datasets": datasets}
        else:
            return {"labels": [],
                    "datasets": [{"name": y_col,
                                  "data": [_safe_float(v) for v in df[y_col].dropna().tolist()]}]}

    return None


def resolve_charts(config, df, top_n, scatter_limit):
    resolved = []
    for ch in config.get("charts", []):
        try:
            result = compute_chart_data(ch, df, top_n=top_n, scatter_limit=scatter_limit)
            if result is None or not result["datasets"]:
                continue
            if not result["labels"] and ch["type"] not in ("histogram", "box"):
                continue
            ch = dict(ch)
            ch["labels"]   = result["labels"]
            ch["datasets"] = result["datasets"]
            resolved.append(ch)
        except Exception as e:
            st.warning(f"Skipped chart '{ch.get('title', '?')}': {e}")
    return resolved


def resolve_metrics(config, df):
    results = []
    for m in config.get("metrics", []):
        label, value = compute_metric(m, df)
        results.append({"label": label, "value": value, "delta": m.get("delta")})
    return results


# ── Chart renderer ─────────────────────────────────────────────────────────────
def render_chart(ch, palette, theme):
    bg, paper, grid, text = theme["bg"], theme["paper"], theme["grid"], theme["text"]

    common_layout = dict(
        paper_bgcolor=paper, plot_bgcolor=bg,
        font=dict(color=text, family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text=ch.get("title", ""), font=dict(size=14, color=text), x=0),
        showlegend=len(ch["datasets"]) > 1,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    axis_style = dict(
        gridcolor=grid, zerolinecolor=grid,
        tickfont=dict(color=text, size=11),
        title_font=dict(color=text),
    )

    ctype  = ch["type"]
    labels = ch.get("labels", [])
    ds     = ch.get("datasets", [])
    fig    = go.Figure()

    if ctype in ("bar", "horizontal_bar"):
        orient = "h" if ctype == "horizontal_bar" else "v"
        for i, d in enumerate(ds):
            x_vals = d["data"] if orient == "h" else labels
            y_vals = labels    if orient == "h" else d["data"]
            fig.add_trace(go.Bar(x=x_vals, y=y_vals, name=d["name"], orientation=orient,
                marker_color=palette[i % len(palette)], marker_line_width=0))
        fig.update_layout(**common_layout, barmode="group" if len(ds) > 1 else "relative")
        if orient == "v":
            fig.update_xaxes(**axis_style, title_text=ch.get("x_label") or "")
            fig.update_yaxes(**axis_style, title_text=ch.get("y_label") or "")
        else:
            fig.update_yaxes(**axis_style, title_text=ch.get("x_label") or "")
            fig.update_xaxes(**axis_style, title_text=ch.get("y_label") or "")

    elif ctype in ("line", "area"):
        fill = "tozeroy" if ctype == "area" else None
        for i, d in enumerate(ds):
            clr = palette[i % len(palette)]
            fig.add_trace(go.Scatter(
                x=labels, y=d["data"], mode="lines+markers", name=d["name"],
                line=dict(color=clr, width=2), marker=dict(size=4),
                fill=fill, fillcolor=clr + "26" if fill else None))
        fig.update_layout(**common_layout)
        fig.update_xaxes(**axis_style, title_text=ch.get("x_label") or "")
        fig.update_yaxes(**axis_style, title_text=ch.get("y_label") or "")

    elif ctype in ("pie", "donut"):
        hole = 0.45 if ctype == "donut" else 0
        fig.add_trace(go.Pie(labels=labels, values=ds[0]["data"] if ds else [],
            hole=hole, marker=dict(colors=palette[:len(labels)], line=dict(color=bg, width=2)),
            textfont=dict(color=text, size=11)))
        fig.update_layout(**common_layout)

    elif ctype == "scatter":
        for i, d in enumerate(ds):
            fig.add_trace(go.Scatter(x=labels, y=d["data"], mode="markers", name=d["name"],
                marker=dict(color=palette[i % len(palette)], size=6, opacity=0.65)))
        fig.update_layout(**common_layout)
        fig.update_xaxes(**axis_style, title_text=ch.get("x_label") or "")
        fig.update_yaxes(**axis_style, title_text=ch.get("y_label") or "")

    elif ctype == "histogram":
        for i, d in enumerate(ds):
            fig.add_trace(go.Histogram(x=d["data"], name=d["name"],
                marker_color=palette[i % len(palette)], nbinsx=30, opacity=0.85))
        fig.update_layout(**common_layout, barmode="overlay")
        fig.update_xaxes(**axis_style, title_text=ch.get("x_label") or "")
        fig.update_yaxes(**axis_style, title_text=ch.get("y_label") or "")

    elif ctype == "box":
        for i, d in enumerate(ds):
            fig.add_trace(go.Box(y=d["data"], name=d["name"],
                marker_color=palette[i % len(palette)], boxpoints="outliers"))
        fig.update_layout(**common_layout)
        fig.update_yaxes(**axis_style, title_text=ch.get("y_label") or "")

    else:
        for i, d in enumerate(ds):
            fig.add_trace(go.Bar(x=labels, y=d["data"], name=d["name"],
                marker_color=palette[i % len(palette)]))
        fig.update_layout(**common_layout)

    return fig


# ── PNG export ─────────────────────────────────────────────────────────────────
def export_dashboard_png(config, resolved_charts, resolved_metrics, df, palette, theme, filename):
    n, cols_n = len(resolved_charts), 2
    rows = math.ceil(n / cols_n)
    W, H_per_row, HEADER_H = 1600, 420, 200

    subplot_types = ["domain" if ch["type"] in ("pie","donut") else "xy" for ch in resolved_charts]
    specs = []
    for r in range(rows):
        row_specs = []
        for c in range(cols_n):
            idx = r * cols_n + c
            row_specs.append({"type": subplot_types[idx] if idx < n else "xy"})
        specs.append(row_specs)

    subtitles = [ch.get("title","") for ch in resolved_charts]
    while len(subtitles) < rows * cols_n:
        subtitles.append("")

    fig = make_subplots(rows=rows, cols=cols_n, subplot_titles=subtitles,
                        specs=specs, horizontal_spacing=0.08, vertical_spacing=0.12)

    for idx, ch in enumerate(resolved_charts):
        r = idx // cols_n + 1
        c = idx % cols_n + 1
        ctype  = ch["type"]
        labels = ch.get("labels", [])
        ds     = ch.get("datasets", [])

        if ctype in ("bar","horizontal_bar"):
            orient = "h" if ctype == "horizontal_bar" else "v"
            for i, d in enumerate(ds):
                xv = d["data"] if orient=="h" else labels
                yv = labels    if orient=="h" else d["data"]
                fig.add_trace(go.Bar(x=xv, y=yv, name=d["name"], orientation=orient,
                    marker_color=palette[i%len(palette)], marker_line_width=0, showlegend=False), row=r, col=c)
        elif ctype in ("line","area"):
            fill = "tozeroy" if ctype=="area" else None
            for i, d in enumerate(ds):
                fig.add_trace(go.Scatter(x=labels, y=d["data"], mode="lines+markers",
                    line=dict(color=palette[i%len(palette)], width=2), marker=dict(size=4),
                    fill=fill, name=d["name"], showlegend=False), row=r, col=c)
        elif ctype in ("pie","donut"):
            hole = 0.4 if ctype=="donut" else 0
            fig.add_trace(go.Pie(labels=labels, values=ds[0]["data"] if ds else [],
                hole=hole, marker=dict(colors=palette[:len(labels)]), showlegend=False), row=r, col=c)
        elif ctype == "scatter":
            for i, d in enumerate(ds):
                fig.add_trace(go.Scatter(x=labels, y=d["data"], mode="markers",
                    marker=dict(color=palette[i%len(palette)], size=5, opacity=0.65),
                    showlegend=False), row=r, col=c)
        elif ctype == "histogram":
            for i, d in enumerate(ds):
                fig.add_trace(go.Histogram(x=d["data"], marker_color=palette[i%len(palette)],
                    nbinsx=30, showlegend=False), row=r, col=c)
        elif ctype == "box":
            for i, d in enumerate(ds):
                fig.add_trace(go.Box(y=d["data"], name=d["name"],
                    marker_color=palette[i%len(palette)], boxpoints="outliers", showlegend=False), row=r, col=c)
        else:
            for i, d in enumerate(ds):
                fig.add_trace(go.Bar(x=labels, y=d["data"],
                    marker_color=palette[i%len(palette)], showlegend=False), row=r, col=c)

    bg, paper, grid, text_color = theme["bg"], theme["paper"], theme["grid"], theme["text"]
    fig.update_layout(paper_bgcolor=paper, plot_bgcolor=bg,
        font=dict(color=text_color, family="Inter, sans-serif", size=11),
        margin=dict(l=60, r=60, t=80, b=40), height=rows*H_per_row, width=W)
    for ann in fig.layout.annotations:
        ann.font.color = text_color
        ann.font.size  = 13
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, tickfont=dict(color=text_color, size=10))
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, tickfont=dict(color=text_color, size=10))

    chart_png = pio.to_image(fig, format="png", width=W, height=rows*H_per_row, scale=2)
    chart_img = Image.open(io.BytesIO(chart_png))

    from PIL import ImageDraw
    header_img = Image.new("RGB", (W*2, HEADER_H*2), color=_hex_to_rgb(bg))
    draw = ImageDraw.Draw(header_img)
    draw.text((80, 60),  config.get("title","Dashboard"),  fill=_hex_to_rgb(text_color), font=_get_font(48, bold=True))
    draw.text((80, 130), f"{len(df):,} rows · {len(df.columns)} columns · {filename}", fill=_hex_to_rgb("#6b7280"), font=_get_font(28))

    metric_x, metric_y = 80, 220
    box_w = (W*2 - 160) // max(len(resolved_metrics), 1) - 20
    for m in resolved_metrics:
        _draw_metric_box(draw, metric_x, metric_y, box_w, 140, m, text_color, bg, palette[0])
        metric_x += box_w + 20

    _draw_insight_box(draw, 80, metric_y+180, W*2-160, config.get("insight",""), text_color, palette[0])

    total_h = header_img.size[1] + chart_img.size[1]
    combined = Image.new("RGB", (W*2, total_h), color=_hex_to_rgb(bg))
    combined.paste(header_img, (0, 0))
    combined.paste(chart_img,  (0, header_img.size[1]))
    buf = io.BytesIO()
    combined.save(buf, format="PNG", dpi=(144, 144))
    return buf.getvalue()


# ── PIL helpers ────────────────────────────────────────────────────────────────
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _get_font(size, bold=False):
    try:
        from PIL import ImageFont
        path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        return ImageFont.truetype(path, size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()

def _draw_metric_box(draw, x, y, w, h, metric, text_color, bg_color, accent):
    r, g, b = _hex_to_rgb(bg_color)
    box_bg = (min(r+20,255), min(g+25,255), min(b+35,255))
    draw.rounded_rectangle([x, y, x+w, y+h], radius=16,
        fill=box_bg, outline=_hex_to_rgb(accent)+(80,))
    draw.text((x+20, y+18),  metric.get("label","")[:30],  fill=(120,130,160),          font=_get_font(22))
    draw.text((x+20, y+55),  metric.get("value","")[:20],  fill=_hex_to_rgb(text_color), font=_get_font(38, bold=True))
    delta = metric.get("delta") or ""
    if delta:
        color = (100,200,130) if "+" in str(delta) else (220,100,100)
        draw.text((x+20, y+105), str(delta), fill=color, font=_get_font(22))

def _draw_insight_box(draw, x, y, w, insight, text_color, accent):
    draw.rectangle([x, y, x+6, y+90], fill=_hex_to_rgb(accent))
    ty = y
    for line in _wrap_text(insight, 115)[:3]:
        draw.text((x+30, ty), line, fill=(160,175,210), font=_get_font(26))
        ty += 36

def _wrap_text(text, max_chars):
    words = text.split()
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line += ("" if not line else " ") + w
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)
    return lines


# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("📊 CSV Auto-Dashboard")
st.caption("AI picks chart types · Python computes all data from your full CSV · export as PNG")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

if not uploaded:
    st.markdown('<div class="upload-prompt">⬆️ Drop a CSV above to get started</div>', unsafe_allow_html=True)
    st.stop()

df = pd.read_csv(uploaded)
uploaded.seek(0)
filename = uploaded.name

c1, c2, c3 = st.columns(3)
c1.metric("Rows",    f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("File",    filename)

with st.expander("📋 Data preview", expanded=False):
    st.dataframe(df.head(20), width="stretch")

st.divider()

if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar. [Get one free →](https://aistudio.google.com/app/apikey)")
    st.stop()

for key in ("dashboard_config", "last_file", "png_bytes"):
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.last_file != filename:
    st.session_state.dashboard_config = None
    st.session_state.png_bytes        = None
    st.session_state.last_file        = filename

if st.session_state.dashboard_config is None:
    with st.spinner("🤖 AI is designing your dashboard layout…"):
        try:
            col_info = infer_column_types(df)
            stats    = compute_summary_stats(df)
            prompt   = build_prompt(df, col_info, stats, filename)
            raw      = call_gemini(api_key, prompt)
            config   = extract_json(raw)
            st.session_state.dashboard_config = config
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.code(str(e))
            st.stop()

config  = st.session_state.dashboard_config
palette = get_palette(chart_theme)

# Compute all chart data and metrics from the FULL DataFrame
resolved_charts  = resolve_charts(config, df, top_n=top_n, scatter_limit=scatter_limit)
resolved_metrics = resolve_metrics(config, df)

# Render
st.subheader(config.get("title", "Dashboard"))

insight = config.get("insight", "")
if insight:
    st.markdown(f'<div class="insight-box">💡 {insight}</div>', unsafe_allow_html=True)

if resolved_metrics:
    cols = st.columns(min(len(resolved_metrics), 6))
    for i, m in enumerate(resolved_metrics):
        cols[i % len(cols)].metric(label=m["label"], value=m["value"], delta=m.get("delta") or None)

st.divider()

chart_cols = st.columns(2)
for idx, ch in enumerate(resolved_charts):
    with chart_cols[idx % 2]:
        try:
            fig = render_chart(ch, palette, theme)
            fig.update_layout(height=340)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            if ch.get("subtitle"):
                st.caption(ch["subtitle"])
        except Exception as e:
            st.warning(f"Could not render '{ch.get('title','chart')}': {e}")

st.divider()

st.subheader("📥 Export dashboard")
col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("🖼️ Generate PNG", type="primary", use_container_width=True):
        with st.spinner("Rendering high-quality PNG…"):
            try:
                png_bytes = export_dashboard_png(
                    config, resolved_charts, resolved_metrics, df, palette, theme, filename)
                st.session_state.png_bytes = png_bytes
                st.success("PNG ready!")
            except Exception as e:
                st.error(f"Export failed: {e}")

if st.session_state.png_bytes:
    with col_b:
        st.download_button(
            label="⬇️ Download PNG",
            data=st.session_state.png_bytes,
            file_name=f"{filename.replace('.csv','')}_dashboard.png",
            mime="image/png",
            use_container_width=True,
        )

st.caption("PNG rendered at 2× resolution (144 DPI) for crisp printing and presentations.")

if st.button("🔄 Regenerate dashboard"):
    st.session_state.dashboard_config = None
    st.session_state.png_bytes        = None
    st.rerun()
