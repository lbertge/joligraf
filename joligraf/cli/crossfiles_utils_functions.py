from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from bokeh.plotting import figure
from bokeh.models import HoverTool


def scale(
    x: ndarray, in_range: Optional[Tuple] = None, out_range: Tuple = (0, 1)
) -> ndarray:
    if in_range is None:
        domain_min, domain_max = np.min(x), np.max(x)
    else:
        domain_min, domain_max = in_range
    a, b = out_range
    return a + ((x - domain_min) * (b - a)) / (domain_max - domain_min)


def make_histogram(title, hist, edges):
    p = figure(title=title, background_fill_color="#fafafa")
    p.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color="navy",
        line_color="white",
        alpha=0.5,
    )

    p.xaxis.axis_label = "x"
    p.yaxis.axis_label = "Pr(x)"
    p.grid.grid_line_color = "white"
    p.add_tools(HoverTool(tooltips=[("count", "@top"), ("bin", "[@left, @right]")]))
    return p
