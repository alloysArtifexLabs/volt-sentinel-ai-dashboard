"""Chart palette and shared Plotly template.

Colours are assigned by the job they do — categorical hues for device
identity, a single-hue blue ramp for magnitude, and a reserved status palette
for anomalies — so no chart ever repurposes a warning colour as "series 4".
The categorical order is validated for colour-vision deficiency separation.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

#: Fixed categorical order. Assign by slot, never cycle.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: Single-hue ramp for continuous magnitude (heatmaps, score scales).
SEQUENTIAL_BLUE: list[str] = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]

#: Reserved for state; always paired with a label or icon, never colour alone.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

SURFACE = "#ffffff"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

TEMPLATE_NAME = "voltsentinel"


def build_template() -> go.layout.Template:
    """Recessive grid, quiet axes, thin marks — the chart chrome for every figure."""
    axis = {
        "showgrid": True,
        "gridcolor": GRIDLINE,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": AXIS,
        "ticks": "outside",
        "tickcolor": AXIS,
        "ticklen": 4,
        "tickfont": {"color": INK_MUTED, "size": 11},
        "title": {"font": {"color": INK_SECONDARY, "size": 12}},
        "automargin": True,
    }
    return go.layout.Template(
        layout={
            "colorway": list(CATEGORICAL),
            "colorscale": {"sequential": [[i / 6, c] for i, c in enumerate(SEQUENTIAL_BLUE)]},
            "font": {"family": FONT_FAMILY, "color": INK_PRIMARY, "size": 13},
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            # The title sits in its own band at the very top; the legend is
            # anchored to the plot area below it, so the two never collide.
            "title": {
                "font": {"size": 15, "color": INK_PRIMARY},
                "x": 0,
                "xanchor": "left",
                "y": 0.97,
                "yanchor": "top",
            },
            "margin": {"l": 8, "r": 8, "t": 92, "b": 8},
            "hoverlabel": {
                "bgcolor": SURFACE,
                "bordercolor": AXIS,
                "font": {"family": FONT_FAMILY, "color": INK_PRIMARY, "size": 12},
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.0,
                "x": 0,
                "title": {"text": ""},
                "font": {"color": INK_SECONDARY, "size": 12},
            },
            "xaxis": axis,
            "yaxis": axis,
        }
    )


def register() -> str:
    """Install the template with Plotly and make it the default."""
    pio.templates[TEMPLATE_NAME] = build_template()
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME


def color_map(categories: list[str]) -> dict[str, str]:
    """Pin each category to a fixed slot.

    Colour follows the entity, so filtering the fleet down never repaints the
    device types that remain.
    """
    return {
        category: CATEGORICAL[index % len(CATEGORICAL)]
        for index, category in enumerate(sorted(categories))
    }
