import plotly.graph_objects as go


def render_chart(ch: dict, palette: list[str], theme: dict) -> go.Figure:
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
        # Fallback: plain bar
        for i, d in enumerate(ds):
            fig.add_trace(go.Bar(x=labels, y=d["data"], name=d["name"],
                marker_color=palette[i % len(palette)]))
        fig.update_layout(**common_layout)

    return fig
