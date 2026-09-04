import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

brg_barve = cm.brg(np.linspace(0, 0.5, 10))
bgr_custom = LinearSegmentedColormap.from_list("bgr_custom", brg_barve)

# Custom Colormaps Setup
gnuplot_barve = cm.gnuplot2_r(np.linspace(0.3, 1, 10))
gnuplot_custom = LinearSegmentedColormap.from_list("gnuplot_custom", gnuplot_barve)

YlGn_barve = cm.YlGn(np.linspace(0, 0.8, 10))
ylgn_custom = LinearSegmentedColormap.from_list("ylgn_custom", YlGn_barve)


MARKERS = ["o", "D", "^", "x", "v", "<", ">", "p", "*", "X"]
SET1_LIST = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#f781bf",
    "#999999",
    "#a65628",
]
BOLD_LIST = [
    "#7f3c8d",
    "#11a579",
    "#3969ac",
    "#e73f74",
    "#f2b701",
    "#80ba5a",
    "#e68310",
    "#008695",
    "#cf1c90",
    "#f97b72",
    "#a5aa99",
]


def apply_plt_style():
    """Applies standard Matplotlib cosmetics."""
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Calibri"],
            "font.size": 8,
            "mathtext.fontset": "cm",
        }
    )

# apply_plt_style()