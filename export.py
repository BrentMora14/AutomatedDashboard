import io
import math

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont


# ── PIL helpers ────────────────────────────────────────────────────────────────

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        path = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        )
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int) -> list[str]:
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


def _draw_metric_box(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    metric: dict, text_color: str, bg_color: str, accent: str,
) -> None:
    r, g, b = _hex_to_rgb(bg_color)
    box_bg = (min(r + 20, 255), min(g + 25, 255), min(b + 35, 255))
    draw.rounded_rectangle(
        [x, y, x + w, y + h], radius=16,
        fill=box_bg, outline=_hex_to_rgb(accent) + (80,),
    )
    draw.text((x + 20, y + 18),  metric.get("label", "")[:30], fill=(120, 130, 160),           font=_get_font(22))
    draw.text((x + 20, y + 55),  metric.get("value", "")[:20], fill=_hex_to_rgb(text_color),   font=_get_font(38, bold=True))
    delta = metric.get("delta") or ""
    if delta:
        color = (100, 200, 130) if "+" in str(delta) else (220, 100, 100)
        draw.text((x + 20, y + 105), str(delta), fill=color, font=_get_font(22))


def _draw_insight_box(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int,
    insight: str, text_color: str, accent: str,
) -> None:
    draw.rectangle([x, y, x + 6, y + 90], fill=_hex_to_rgb(accent))
    ty = y
    for line in _wrap_text(insight, 115)[:3]:
        draw.text((x + 30, ty), line, fill=(160, 175, 210), font=_get_font(26))
        ty += 36


# ── Subplot figure builder ─────────────────────────────────────────────────────

def _build_subplot_figure(
    resolved_charts: list[dict],
    palette: list[str],
    theme: dict,
    rows: int,
    cols_n: int,
    W: int,
    H_per_row: int,
) -> go.Figure:
    """Build a multi-panel Plotly figure for PNG export."""
    n = len(resolved_charts)
    bg, paper, grid, text_color = theme["bg"], theme["paper"], theme["grid"], theme["text"]

    subplot_types = [
        "domain" if ch["type"] in ("pie", "donut") else "xy"
        for ch in resolved_charts
    ]
    specs = []
    for r in range(rows):
        row_specs = []
        for c in range(cols_n):
            idx = r * cols_n + c
            row_specs.append({"type": subplot_types[idx] if idx < n else "xy"})
        specs.append(row_specs)

    subtitles = [ch.get("title", "") for ch in resolved_charts]
    while len(subtitles) < rows * cols_n:
        subtitles.append("")

    fig = make_subplots(
        rows=rows, cols=cols_n,
        subplot_titles=subtitles,
        specs=specs,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, ch in enumerate(resolved_charts):
        r = idx // cols_n + 1
        c = idx % cols_n + 1
        ctype  = ch["type"]
        labels = ch.get("labels", [])
        ds     = ch.get("datasets", [])

        if ctype in ("bar", "horizontal_bar"):
            orient = "h" if ctype == "horizontal_bar" else "v"
            for i, d in enumerate(ds):
                xv = d["data"] if orient == "h" else labels
                yv = labels    if orient == "h" else d["data"]
                fig.add_trace(go.Bar(
                    x=xv, y=yv, name=d["name"], orientation=orient,
                    marker_color=palette[i % len(palette)], marker_line_width=0,
                    showlegend=False,
                ), row=r, col=c)

        elif ctype in ("line", "area"):
            fill = "tozeroy" if ctype == "area" else None
            for i, d in enumerate(ds):
                fig.add_trace(go.Scatter(
                    x=labels, y=d["data"], mode="lines+markers",
                    line=dict(color=palette[i % len(palette)], width=2),
                    marker=dict(size=4), fill=fill,
                    name=d["name"], showlegend=False,
                ), row=r, col=c)

        elif ctype in ("pie", "donut"):
            hole = 0.4 if ctype == "donut" else 0
            fig.add_trace(go.Pie(
                labels=labels, values=ds[0]["data"] if ds else [],
                hole=hole, marker=dict(colors=palette[:len(labels)]),
                showlegend=False,
            ), row=r, col=c)

        elif ctype == "scatter":
            for i, d in enumerate(ds):
                fig.add_trace(go.Scatter(
                    x=labels, y=d["data"], mode="markers",
                    marker=dict(color=palette[i % len(palette)], size=5, opacity=0.65),
                    showlegend=False,
                ), row=r, col=c)

        elif ctype == "histogram":
            for i, d in enumerate(ds):
                fig.add_trace(go.Histogram(
                    x=d["data"], marker_color=palette[i % len(palette)],
                    nbinsx=30, showlegend=False,
                ), row=r, col=c)

        elif ctype == "box":
            for i, d in enumerate(ds):
                fig.add_trace(go.Box(
                    y=d["data"], name=d["name"],
                    marker_color=palette[i % len(palette)],
                    boxpoints="outliers", showlegend=False,
                ), row=r, col=c)

        else:
            for i, d in enumerate(ds):
                fig.add_trace(go.Bar(
                    x=labels, y=d["data"],
                    marker_color=palette[i % len(palette)],
                    showlegend=False,
                ), row=r, col=c)

    fig.update_layout(
        paper_bgcolor=paper, plot_bgcolor=bg,
        font=dict(color=text_color, family="Inter, sans-serif", size=11),
        margin=dict(l=60, r=60, t=80, b=40),
        height=rows * H_per_row,
        width=W,
    )
    for ann in fig.layout.annotations:
        ann.font.color = text_color
        ann.font.size  = 13
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, tickfont=dict(color=text_color, size=10))
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, tickfont=dict(color=text_color, size=10))

    return fig


# ── Main export entry point ────────────────────────────────────────────────────

def export_dashboard_png(
    config: dict,
    resolved_charts: list[dict],
    resolved_metrics: list[dict],
    df,
    palette: list[str],
    theme: dict,
    filename: str,
) -> bytes:
    cols_n, W, H_per_row, HEADER_H = 2, 1600, 420, 200
    n    = len(resolved_charts)
    rows = math.ceil(n / cols_n)

    bg, text_color = theme["bg"], theme["text"]

    # ── Render chart grid as PNG ───────────────────────────────────────────────
    fig = _build_subplot_figure(resolved_charts, palette, theme, rows, cols_n, W, H_per_row)
    chart_png = pio.to_image(fig, format="png", width=W, height=rows * H_per_row, scale=2)
    chart_img = Image.open(io.BytesIO(chart_png))

    # ── Build header image ─────────────────────────────────────────────────────
    header_img = Image.new("RGB", (W * 2, HEADER_H * 2), color=_hex_to_rgb(bg))
    draw = ImageDraw.Draw(header_img)

    draw.text(
        (80, 60), config.get("title", "Dashboard"),
        fill=_hex_to_rgb(text_color), font=_get_font(48, bold=True),
    )
    draw.text(
        (80, 130), f"{len(df):,} rows · {len(df.columns)} columns · {filename}",
        fill=_hex_to_rgb("#6b7280"), font=_get_font(28),
    )

    # Metric boxes
    metric_x, metric_y = 80, 220
    box_w = (W * 2 - 160) // max(len(resolved_metrics), 1) - 20
    for m in resolved_metrics:
        _draw_metric_box(draw, metric_x, metric_y, box_w, 140, m, text_color, bg, palette[0])
        metric_x += box_w + 20

    # Insight strip
    _draw_insight_box(draw, 80, metric_y + 180, W * 2 - 160,
                      config.get("insight", ""), text_color, palette[0])

    # ── Stitch header + charts ─────────────────────────────────────────────────
    total_h = header_img.size[1] + chart_img.size[1]
    combined = Image.new("RGB", (W * 2, total_h), color=_hex_to_rgb(bg))
    combined.paste(header_img, (0, 0))
    combined.paste(chart_img,  (0, header_img.size[1]))

    buf = io.BytesIO()
    combined.save(buf, format="PNG", dpi=(144, 144))
    return buf.getvalue()
