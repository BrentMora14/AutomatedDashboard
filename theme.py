THEMES = {
    "Dark ocean":    {"bg": "#0f1117", "paper": "#161b27", "grid": "#1e2535", "text": "#c8d0e8"},
    "Midnight blue": {"bg": "#060c1e", "paper": "#0d1530", "grid": "#1a2545", "text": "#b8c8f0"},
    "Forest green":  {"bg": "#0b1209", "paper": "#121f10", "grid": "#1a2f18", "text": "#c0d8b8"},
    "Warm ember":    {"bg": "#130c05", "paper": "#1f1208", "grid": "#2e1c0a", "text": "#e8d0b0"},
}

PALETTES = {
    "Dark ocean":    ["#4e8af4","#34c7a0","#f4844e","#c97cf4","#f4c94e","#f4506e","#4ec9f4"],
    "Midnight blue": ["#6b9ff5","#5ec8e8","#c589f5","#f58c6b","#f5d06b","#5ef5b4","#f56b8c"],
    "Forest green":  ["#5ec47a","#a0d45e","#5eb8a0","#d4a05e","#5e8ad4","#d45e7a","#c4d45e"],
    "Warm ember":    ["#f4844e","#f4c94e","#f4506e","#f4a04e","#c4784e","#f4e44e","#e87850"],
}

THEME_NAMES = list(THEMES.keys())


def get_palette(theme_name: str) -> list[str]:
    return PALETTES.get(theme_name, PALETTES["Dark ocean"])


def get_theme(theme_name: str) -> dict:
    return THEMES.get(theme_name, THEMES["Dark ocean"])
