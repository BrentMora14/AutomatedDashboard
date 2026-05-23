import pandas as pd

AGG_MAP = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}


def _safe_float(v) -> float:
    try:
        return round(float(v), 6)
    except Exception:
        return 0.0


# ── Schema analysis ────────────────────────────────────────────────────────────

def infer_column_types(df: pd.DataFrame) -> dict:
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


def compute_summary_stats(df: pd.DataFrame) -> dict:
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


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metric(spec: dict, df: pd.DataFrame) -> tuple[str, str]:
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


def resolve_metrics(config: dict, df: pd.DataFrame) -> list[dict]:
    results = []
    for m in config.get("metrics", []):
        label, value = compute_metric(m, df)
        results.append({"label": label, "value": value, "delta": m.get("delta")})
    return results


# ── Chart data computation ─────────────────────────────────────────────────────

def compute_chart_data(spec: dict, df: pd.DataFrame, top_n: int = 15, scatter_limit: int = 500) -> dict | None:
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


def resolve_charts(config: dict, df: pd.DataFrame, top_n: int, scatter_limit: int) -> list[dict]:
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
            # Caller (app.py) is responsible for surfacing warnings to the UI
            print(f"[data] Skipped chart '{ch.get('title', '?')}': {e}")
    return resolved
