"""Consistent, quiet chart styling for the notebooks."""
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLORS = {"persistence": "#9b9a95", "linear": "#2a78d6", "gbm": "#eb6834", "lstm": "#1baf7a",
                "tuned": "#6250d6"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def set_style():
    plt.rcParams.update({
        "figure.figsize": (9, 4.5), "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
        "axes.titlelocation": "left", "axes.titlesize": 11, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "font.size": 9.5,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def model_color(name):
    n = name.lower()
    for k, c in MODEL_COLORS.items():
        if n.startswith(k):
            return c
    return MUTED
