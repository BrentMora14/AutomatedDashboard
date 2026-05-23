import json
import re

from google import genai
from google.genai import types as genai_types


def build_prompt(df, col_info: dict, stats: dict, filename: str) -> str:
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


def call_gemini(api_keys: list[str], prompt: str) -> str:
    """Try each API key in order, falling back to the next on failure."""
    last_error = None
    for key in api_keys:
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text
        except Exception as e:
            last_error = e
            continue
    raise last_error


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    return json.loads(text)


def get_dashboard_config(api_keys: list[str], df, col_info: dict, stats: dict, filename: str) -> dict:
    """Full pipeline: build prompt → call Gemini (with key fallback) → parse JSON. Raises on failure."""
    prompt = build_prompt(df, col_info, stats, filename)
    raw    = call_gemini(api_keys, prompt)
    return extract_json(raw)


def build_additional_chart_prompt(user_prompt: str, col_info: dict, stats: dict, filename: str) -> str:
    return f"""You are an expert data analyst. A user wants to add one custom chart to an existing dashboard.
Return ONLY a single valid JSON chart spec — no explanation, no markdown, no extra text.

Dataset: {filename}
Column types: {json.dumps(col_info)}
Numeric stats: {json.dumps(stats, indent=2)}

User request: "{user_prompt}"

Return this exact JSON structure:
{{
  "id": "custom_<short_unique_id>",
  "type": "bar|horizontal_bar|line|area|pie|donut|scatter|histogram|box",
  "title": "descriptive chart title",
  "subtitle": "one sentence explaining what this shows",
  "x_col": "exact column name or null",
  "y_col": "exact column name or null",
  "col": "exact column name (histogram only) or null",
  "agg": "sum|mean|count|min|max",
  "sort": "desc|asc|none",
  "limit": 15,
  "time_group": "day|month|year|null",
  "x_label": "human-readable x label or null",
  "y_label": "human-readable y label or null",
  "filters": [
    {{
      "col": "exact column name to filter on",
      "op": "eq|ne|gt|lt|gte|lte|in|contains|month_eq|year_eq",
      "value": "scalar value, or list of values for op=in"
    }}
  ]
}}

Filter op reference:
- month_eq / year_eq: value is an integer (e.g. 5 for May, 2024 for year)
- eq / ne / gt / lt / gte / lte: direct value comparison
- in: value must be a JSON array
- contains: case-insensitive substring match on string columns

Use an empty list [] for filters if no filtering is needed.
Only use column names that exist in the dataset. Return ONLY the JSON object."""


def get_additional_chart(api_keys: list[str], user_prompt: str, col_info: dict, stats: dict, filename: str) -> dict:
    """Translate a natural language chart request into a single chart spec. Raises on failure."""
    prompt = build_additional_chart_prompt(user_prompt, col_info, stats, filename)
    raw    = call_gemini(api_keys, prompt)
    return extract_json(raw)