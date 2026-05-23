import streamlit as st
import pandas as pd

from theme import THEME_NAMES, get_palette, get_theme
from data import infer_column_types, compute_summary_stats, resolve_charts, resolve_metrics
from gemini import get_dashboard_config
from charts import render_chart
from export import export_dashboard_png

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

# ── Secrets ────────────────────────────────────────────────────────────────────
api_key = st.secrets.get("GEMINI_API_KEY", "")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()
    chart_theme   = st.selectbox("Color theme", THEME_NAMES)
    top_n         = st.slider("Top-N categories per chart", 5, 30, 15, step=5,
                        help="Max unique category values shown per chart")
    scatter_limit = st.slider("Scatter point limit", 100, 2000, 500, step=100,
                        help="Max rows plotted on scatter charts (for performance)")
    st.divider()
    st.markdown("**About**")
    st.caption("AI picks chart types & columns. Python computes all data from your full CSV.")

palette = get_palette(chart_theme)
theme   = get_theme(chart_theme)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 CSV Auto-Dashboard")
st.caption("AI picks chart types · Python computes all data from your full CSV · export as PNG")

# ── File upload ────────────────────────────────────────────────────────────────
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

# ── API key guard ──────────────────────────────────────────────────────────────
if not api_key:
    st.error("⚠️ Gemini API key not configured. Add GEMINI_API_KEY to the app's Streamlit Cloud secrets.")
    st.stop()

# ── Session state init ─────────────────────────────────────────────────────────
for key in ("dashboard_config", "last_file", "png_bytes"):
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.last_file != filename:
    st.session_state.dashboard_config = None
    st.session_state.png_bytes        = None
    st.session_state.last_file        = filename

# ── AI layout generation ───────────────────────────────────────────────────────
if st.session_state.dashboard_config is None:
    with st.spinner("🤖 AI is designing your dashboard layout…"):
        try:
            col_info = infer_column_types(df)
            stats    = compute_summary_stats(df)
            config   = get_dashboard_config(api_key, df, col_info, stats, filename)
            st.session_state.dashboard_config = config
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.code(str(e))
            st.stop()

config = st.session_state.dashboard_config

# ── Data resolution (always uses full DataFrame) ───────────────────────────────
resolved_charts  = resolve_charts(config, df, top_n=top_n, scatter_limit=scatter_limit)
resolved_metrics = resolve_metrics(config, df)

# ── Dashboard render ───────────────────────────────────────────────────────────
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
            st.warning(f"Could not render '{ch.get('title', 'chart')}': {e}")

st.divider()

# ── PNG export ─────────────────────────────────────────────────────────────────
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
            file_name=f"{filename.replace('.csv', '')}_dashboard.png",
            mime="image/png",
            use_container_width=True,
        )

st.caption("PNG rendered at 2× resolution (144 DPI) for crisp printing and presentations.")

if st.button("🔄 Regenerate dashboard"):
    st.session_state.dashboard_config = None
    st.session_state.png_bytes        = None
    st.rerun()
