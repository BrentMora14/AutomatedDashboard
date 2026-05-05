import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import json
import re
import io
import math
from PIL import Image
import google.generativeai as genai

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
    .metric-container { background: #161b27; border: 1px solid #2a2f45; border-radius: 10px; padding: 1rem 1.25rem; }
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

PALETTE = [
    "#4e8af4", "#34c7a0", "#f4844e", "#c97cf4", "#f4c94e",
    "#f4506e", "#4ec9f4", "#a0f48a", "#f4a04e", "#8a4ef4",
]

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
    max_rows_preview = st.slider("Rows sent to AI", 50, 300, 120, step=10,
        help="More rows = better analysis, slower response")
    st.divider()
    st.markdown("**About**")
    st.caption("Upload any CSV → AI auto-detects columns, picks chart types, and builds a dashboard. Export as PNG.")

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

# ── Helpers ────────────────────────────────────────────────────────────────────
def infer_column_types(df):
    col_info = {}
    for col in df.columns:
        s = df[col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            col_info[col] = "numeric"
        else:
            try:
                pd.to_datetime(s.head(20), infer_datetime_format=True)
                col_info[col] = "datetime"
            except Exception:
                n_unique = s.nunique()
                col_info[col] = f"categorical ({n_unique} unique)"
    return col_info

def compute_summary_stats(df):
    stats = {}
    for col in df.select_dtypes(include="number").columns:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": round(float(df[col].mean()), 4),
            "sum": float(df[col].sum()),
        }
    return stats

def call_gemini(api_key, prompt):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

def extract_json(text):
    text = text.strip()
    # strip markdown fences
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # find outermost { }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    return json.loads(text)

def build_prompt(df, col_info, stats, filename, n_preview):
    preview = df.head(n_preview).to_dict(orient="records")
    return f"""You are an expert data analyst. Analyze this CSV and return ONLY valid JSON — no markdown, no explanation, no extra text.

Filename: {filename}
Total rows: {len(df)}
Columns and types: {json.dumps(col_info, indent=2)}
Summary statistics: {json.dumps(stats, indent=2)}
Sample data ({n_preview} rows): {json.dumps(preview, default=str)}

Return this exact JSON structure:
{{
  "title": "3-6 word dashboard title",
  "insight": "3-4 sentence executive summary — key patterns, anomalies, trends, or notable findings a manager would care about",
  "metrics": [
    {{ "label": "metric name", "value": "formatted value", "delta": "optional change e.g. +12%" }}
  ],
  "charts": [
    {{
      "id": "unique_id",
      "type": "bar|horizontal_bar|line|area|pie|donut|scatter|histogram|box",
      "title": "chart title",
      "subtitle": "one sentence explaining what this shows",
      "x_label": "axis label or null",
      "y_label": "axis label or null",
      "labels": ["label1", ...],
      "datasets": [
        {{ "name": "series name", "data": [1, 2, 3, ...] }}
      ]
    }}
  ]
}}

Rules:
- metrics: 4-6 KPIs (totals, averages, counts, min/max, ratios). Format values nicely (e.g. "$1.2M", "3,421", "87.4%").
- charts: 4-7 charts. Choose types that genuinely suit the data:
  * bar/horizontal_bar → categories, rankings, comparisons
  * line/area → time series, trends over ordered axis
  * pie/donut → parts of a whole (use only when ≤8 categories)
  * scatter → correlations between two numeric columns
  * histogram → distribution of a numeric column
  * box → spread/outliers of numeric data grouped by category
- For time/date columns: parse and sort chronologically, use as x-axis for line/area charts.
- For categories with many values: show top 10 by count or sum.
- labels and every data array must have IDENTICAL length.
- All data values must be numbers (no strings, no nulls).
- Derive everything from the actual data — never invent values.
- Return ONLY the JSON object."""

# ── Chart renderer ──────────────────────────────────────────────────────────────
def render_chart(ch, palette, theme):
    bg    = theme["bg"]
    paper = theme["paper"]
    grid  = theme["grid"]
    text  = theme["text"]

    common_layout = dict(
        paper_bgcolor=paper,
        plot_bgcolor=bg,
        font=dict(color=text, family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text=ch["title"], font=dict(size=14, color=text), x=0),
        showlegend=len(ch["datasets"]) > 1,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    axis_style = dict(
        gridcolor=grid, zerolinecolor=grid,
        tickfont=dict(color=text, size=11),
        titlefont=dict(color=text),
    )

    ctype = ch["type"]
    labels = ch.get("labels", [])
    ds = ch.get("datasets", [])

    fig = go.Figure()

    if ctype in ("bar", "horizontal_bar"):
        orient = "h" if ctype == "horizontal_bar" else "v"
        for i, d in enumerate(ds):
            x_vals = d["data"] if orient == "h" else labels
            y_vals = labels if orient == "h" else d["data"]
            fig.add_trace(go.Bar(
                x=x_vals, y=y_vals,
                name=d["name"], orientation=orient,
                marker_color=palette[i % len(palette)],
                marker_line_width=0,
            ))
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
            fig.add_trace(go.Scatter(
                x=labels, y=d["data"],
                mode="lines+markers",
                name=d["name"],
                line=dict(color=palette[i % len(palette)], width=2),
                marker=dict(size=4),
                fill=fill,
                fillcolor=palette[i % len(palette)].replace("#", "rgba(") + ",0.15)" if fill else None,
            ))
        fig.update_layout(**common_layout)
        fig.update_xaxes(**axis_style, title_text=ch.get("x_label") or "")
        fig.update_yaxes(**axis_style, title_text=ch.get("y_label") or "")

    elif ctype in ("pie", "donut"):
        hole = 0.45 if ctype == "donut" else 0
        fig.add_trace(go.Pie(
            labels=labels,
            values=ds[0]["data"] if ds else [],
            hole=hole,
            marker=dict(colors=palette[:len(labels)], line=dict(color=bg, width=2)),
            textfont=dict(color=text, size=11),
        ))
        fig.update_layout(**common_layout)

    elif ctype == "scatter":
        for i, d in enumerate(ds):
            x_data = labels if labels else list(range(len(d["data"])))
            fig.add_trace(go.Scatter(
                x=x_data, y=d["data"],
                mode="markers",
                name=d["name"],
                marker=dict(color=palette[i % len(palette)], size=7, opacity=0.75),
            ))
        fig.update_layout(**common_layout)
        fig.update_xaxes(**axis_style, title_text=ch.get("x_label") or "")
        fig.update_yaxes(**axis_style, title_text=ch.get("y_label") or "")

    elif ctype == "histogram":
        for i, d in enumerate(ds):
            fig.add_trace(go.Histogram(
                x=d["data"], name=d["name"],
                marker_color=palette[i % len(palette)],
                nbinsx=20, opacity=0.85,
            ))
        fig.update_layout(**common_layout, barmode="overlay")
        fig.update_xaxes(**axis_style, title_text=ch.get("x_label") or "")
        fig.update_yaxes(**axis_style, title_text=ch.get("y_label") or "")

    elif ctype == "box":
        for i, d in enumerate(ds):
            fig.add_trace(go.Box(
                y=d["data"], name=d["name"],
                marker_color=palette[i % len(palette)],
                boxpoints="outliers",
            ))
        fig.update_layout(**common_layout)
        fig.update_yaxes(**axis_style, title_text=ch.get("y_label") or "")

    else:
        # fallback to bar
        for i, d in enumerate(ds):
            fig.add_trace(go.Bar(x=labels, y=d["data"], name=d["name"], marker_color=palette[i % len(palette)]))
        fig.update_layout(**common_layout)

    return fig


# ── PNG export ─────────────────────────────────────────────────────────────────
def export_dashboard_png(config, df, palette, theme, filename):
    charts = config.get("charts", [])
    metrics = config.get("metrics", [])
    n = len(charts)
    cols = 2
    rows = math.ceil(n / cols)

    W, H_per_row = 1600, 420
    HEADER_H = 200
    total_h = HEADER_H + rows * H_per_row + 60

    # Build combined figure with subplots
    from plotly.subplots import make_subplots as _make_subplots
    subplot_types = []
    for ch in charts:
        if ch["type"] in ("pie", "donut"):
            subplot_types.append("domain")
        else:
            subplot_types.append("xy")

    specs = []
    for r in range(rows):
        row_specs = []
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                row_specs.append({"type": subplot_types[idx]})
            else:
                row_specs.append({"type": "xy"})
        specs.append(row_specs)

    subtitles = [ch["title"] for ch in charts]
    while len(subtitles) < rows * cols:
        subtitles.append("")

    fig = _make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subtitles,
        specs=specs,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, ch in enumerate(charts):
        r = idx // cols + 1
        c = idx % cols + 1
        ctype = ch["type"]
        labels = ch.get("labels", [])
        ds = ch.get("datasets", [])

        if ctype in ("bar", "horizontal_bar"):
            orient = "h" if ctype == "horizontal_bar" else "v"
            for i, d in enumerate(ds):
                xv = d["data"] if orient == "h" else labels
                yv = labels if orient == "h" else d["data"]
                fig.add_trace(go.Bar(x=xv, y=yv, name=d["name"],
                    orientation=orient, marker_color=palette[i % len(palette)],
                    marker_line_width=0, showlegend=False), row=r, col=c)
        elif ctype in ("line", "area"):
            fill = "tozeroy" if ctype == "area" else None
            for i, d in enumerate(ds):
                fig.add_trace(go.Scatter(
                    x=labels, y=d["data"], mode="lines+markers",
                    line=dict(color=palette[i % len(palette)], width=2),
                    marker=dict(size=4),
                    fill=fill, name=d["name"], showlegend=False), row=r, col=c)
        elif ctype in ("pie", "donut"):
            hole = 0.4 if ctype == "donut" else 0
            fig.add_trace(go.Pie(
                labels=labels, values=ds[0]["data"] if ds else [],
                hole=hole, marker=dict(colors=palette[:len(labels)]),
                showlegend=False), row=r, col=c)
        elif ctype == "scatter":
            xv = labels if labels else list(range(len(ds[0]["data"]))) if ds else []
            for i, d in enumerate(ds):
                fig.add_trace(go.Scatter(
                    x=xv, y=d["data"], mode="markers",
                    marker=dict(color=palette[i % len(palette)], size=6),
                    showlegend=False), row=r, col=c)
        elif ctype == "histogram":
            for i, d in enumerate(ds):
                fig.add_trace(go.Histogram(
                    x=d["data"], marker_color=palette[i % len(palette)],
                    nbinsx=20, showlegend=False), row=r, col=c)
        else:
            for i, d in enumerate(ds):
                fig.add_trace(go.Bar(x=labels, y=d["data"],
                    marker_color=palette[i % len(palette)], showlegend=False), row=r, col=c)

    bg = theme["bg"]
    paper = theme["paper"]
    grid = theme["grid"]
    text_color = theme["text"]

    fig.update_layout(
        paper_bgcolor=paper,
        plot_bgcolor=bg,
        font=dict(color=text_color, family="Inter, sans-serif", size=11),
        margin=dict(l=60, r=60, t=80, b=40),
        height=rows * H_per_row,
        width=W,
    )
    for ann in fig.layout.annotations:
        ann.font.color = text_color
        ann.font.size = 13

    axis_upd = dict(gridcolor=grid, zerolinecolor=grid,
                    tickfont=dict(color=text_color, size=10))
    fig.update_xaxes(**axis_upd)
    fig.update_yaxes(**axis_upd)

    # Export charts section as PNG bytes
    chart_png_bytes = pio.to_image(fig, format="png", width=W, height=rows * H_per_row, scale=2)
    chart_img = Image.open(io.BytesIO(chart_png_bytes))

    # Build header image with PIL
    from PIL import ImageDraw, ImageFont
    header_img = Image.new("RGB", (W * 2, HEADER_H * 2), color=_hex_to_rgb(bg))
    draw = ImageDraw.Draw(header_img)

    # Title
    title_text = config.get("title", "Dashboard")
    draw.text((80, 60), title_text, fill=_hex_to_rgb(text_color), font=_get_font(48, bold=True))
    # Subtitle
    sub = f"{len(df):,} rows · {len(df.columns)} columns · {filename}"
    draw.text((80, 130), sub, fill=_hex_to_rgb("#6b7280"), font=_get_font(28))

    # Metrics row
    metric_x = 80
    metric_y = 200
    box_w = (W * 2 - 160) // max(len(metrics), 1) - 20
    box_h = 140
    for m in metrics:
        _draw_metric_box(draw, metric_x, metric_y, box_w, box_h, m, text_color, bg, palette[0])
        metric_x += box_w + 20

    # Insight box
    insight_y = metric_y + box_h + 40
    insight_text = config.get("insight", "")
    _draw_insight_box(draw, 80, insight_y, W * 2 - 160, insight_text, text_color, palette[0])

    total_h_px = header_img.size[1] + chart_img.size[1]
    combined = Image.new("RGB", (W * 2, total_h_px), color=_hex_to_rgb(bg))
    combined.paste(header_img, (0, 0))
    combined.paste(chart_img, (0, header_img.size[1]))

    buf = io.BytesIO()
    combined.save(buf, format="PNG", dpi=(144, 144))
    return buf.getvalue()


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _get_font(size, bold=False):
    try:
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        from PIL import ImageFont
        return ImageFont.truetype(path, size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()

def _draw_metric_box(draw, x, y, w, h, metric, text_color, bg_color, accent):
    r, g, b = _hex_to_rgb(bg_color)
    box_bg = (min(r+20, 255), min(g+25, 255), min(b+35, 255))
    draw.rounded_rectangle([x, y, x+w, y+h], radius=16, fill=box_bg, outline=_hex_to_rgb(accent)+(80,))
    label = metric.get("label", "")
    value = metric.get("value", "")
    delta = metric.get("delta", "")
    draw.text((x+20, y+20), label[:30], fill=(120, 130, 160), font=_get_font(22))
    draw.text((x+20, y+55), value[:20], fill=_hex_to_rgb(text_color), font=_get_font(38, bold=True))
    if delta:
        color = (100, 200, 130) if "+" in delta else (220, 100, 100)
        draw.text((x+20, y+105), delta, fill=color, font=_get_font(22))

def _draw_insight_box(draw, x, y, w, insight, text_color, accent):
    draw.rectangle([x, y, x+6, y+80], fill=_hex_to_rgb(accent))
    lines = _wrap_text(insight, 110)
    ty = y
    for line in lines[:3]:
        draw.text((x+30, ty), line, fill=(160, 175, 210), font=_get_font(26))
        ty += 36

def _wrap_text(text, max_chars):
    words = text.split()
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line += ("" if not line else " ") + w
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("📊 CSV Auto-Dashboard")
st.caption("Upload a CSV → AI analyzes it → instant dashboard + PNG export")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

if not uploaded:
    st.markdown('<div class="upload-prompt">⬆️ Drop a CSV above to get started</div>', unsafe_allow_html=True)
    st.stop()

# Load CSV
df = pd.read_csv(uploaded)
uploaded.seek(0)
filename = uploaded.name

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("File", filename)

with st.expander("📋 Data preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

if not api_key:
    st.info("👈 Enter your free Gemini API key in the sidebar to generate the dashboard. [Get one free →](https://aistudio.google.com/app/apikey)")
    st.stop()

if "dashboard_config" not in st.session_state:
    st.session_state.dashboard_config = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None

if st.session_state.last_file != filename:
    st.session_state.dashboard_config = None
    st.session_state.last_file = filename

if st.session_state.dashboard_config is None:
    with st.spinner("🤖 AI is analyzing your data and designing the dashboard…"):
        try:
            col_info = infer_column_types(df)
            stats = compute_summary_stats(df)
            prompt = build_prompt(df, col_info, stats, filename, max_rows_preview)
            raw = call_gemini(api_key, prompt)
            config = extract_json(raw)
            st.session_state.dashboard_config = config
        except Exception as e:
            st.error(f"❌ Error generating dashboard: {e}")
            st.code(str(e))
            st.stop()

config = st.session_state.dashboard_config
palette = get_palette(chart_theme)

# ── Render dashboard ───────────────────────────────────────────────────────────
st.subheader(config.get("title", "Dashboard"))

insight = config.get("insight", "")
if insight:
    st.markdown(f'<div class="insight-box">💡 {insight}</div>', unsafe_allow_html=True)

# Metrics row
metrics = config.get("metrics", [])
if metrics:
    cols = st.columns(min(len(metrics), 6))
    for i, m in enumerate(metrics):
        delta_val = m.get("delta") or None
        cols[i % len(cols)].metric(label=m["label"], value=m["value"], delta=delta_val)

st.divider()

# Charts grid
charts = config.get("charts", [])
n_charts = len(charts)
chart_cols = st.columns(2)

for idx, ch in enumerate(charts):
    with chart_cols[idx % 2]:
        try:
            fig = render_chart(ch, palette, theme)
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            if ch.get("subtitle"):
                st.caption(ch["subtitle"])
        except Exception as e:
            st.warning(f"Could not render '{ch.get('title', 'chart')}': {e}")

st.divider()

# ── Export section ─────────────────────────────────────────────────────────────
st.subheader("📥 Export dashboard")

col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("🖼️ Generate PNG", type="primary", use_container_width=True):
        with st.spinner("Rendering high-quality PNG…"):
            try:
                png_bytes = export_dashboard_png(config, df, palette, theme, filename)
                st.session_state.png_bytes = png_bytes
                st.success("PNG ready!")
            except Exception as e:
                st.error(f"Export failed: {e}")

if "png_bytes" in st.session_state:
    with col_b:
        st.download_button(
            label="⬇️ Download PNG",
            data=st.session_state.png_bytes,
            file_name=f"{filename.replace('.csv','')}_dashboard.png",
            mime="image/png",
            use_container_width=True,
        )

st.caption("PNG is rendered at 2× resolution (144 DPI) for crisp printing and presentations.")

# Regenerate button
if st.button("🔄 Regenerate dashboard", use_container_width=False):
    st.session_state.dashboard_config = None
    if "png_bytes" in st.session_state:
        del st.session_state.png_bytes
    st.rerun()
