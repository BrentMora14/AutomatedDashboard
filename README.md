# 📊 CSV Auto-Dashboard

Upload any CSV file and instantly get an AI-generated dashboard with relevant charts, key metrics, and an executive summary — all exportable as a high-quality PNG.

---

## ✨ Features

- **Auto-analysis** — Gemini AI detects column types, patterns, and relationships
- **Smart chart selection** — bar, line, area, pie, donut, scatter, histogram, box plots
- **Key metrics** — auto-computed KPIs (totals, averages, ratios, extremes)
- **Executive insight** — 3–4 sentence AI summary of what the data shows
- **PNG export** — 2× resolution (144 DPI), ready for presentations or reports
- **Color themes** — Dark ocean, Midnight blue, Forest green, Warm ember
- **Completely free** — uses Google Gemini 2.0 Flash (free tier, no credit card)

---

## 🚀 Quick Start

### 1. Get a free Gemini API key
Go to → https://aistudio.google.com/app/apikey  
Click **"Create API key"** — no credit card required.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Use it
1. Paste your Gemini API key in the sidebar
2. Upload any `.csv` file
3. Watch the dashboard generate automatically
4. Click **"Generate PNG"** → **"Download PNG"** to export

---

## 📁 Project Structure

```
csv_dashboard/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## 🧠 How It Works

1. **CSV parsing** — pandas reads your file and infers column types (numeric, categorical, datetime)
2. **Summary stats** — min, max, mean, sum computed per numeric column
3. **AI prompt** — a structured prompt with sample rows + stats is sent to Gemini 2.0 Flash
4. **JSON config** — Gemini returns a dashboard spec: title, insight, metrics, and chart configs
5. **Plotly rendering** — charts are rendered interactively in Streamlit
6. **PNG export** — Plotly subplots + PIL header are composited into a high-res image

---

## 💡 Tips

- Works best with **structured, tabular CSVs** (sales data, analytics exports, survey results, financials)
- Increase **"Rows sent to AI"** slider for more accurate analysis on large files
- Hit **"Regenerate dashboard"** to get a fresh AI interpretation
- Change the **color theme** in the sidebar before exporting PNG

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` | CSV parsing & stats |
| `plotly` | Interactive charts & PNG rendering |
| `google-generativeai` | Gemini AI API (free) |
| `Pillow` | PNG compositing (header, metrics, insight) |

---

## 🔑 API Usage & Limits

Gemini 2.0 Flash free tier (as of 2025):
- **15 requests/minute**
- **1,500 requests/day**
- **1M tokens/minute**

This is more than enough for personal or small-team use.
