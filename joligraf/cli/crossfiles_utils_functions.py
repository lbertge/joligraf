from typing import Optional, Tuple

import numpy as np
from numpy import ndarray


def scale(
    x: ndarray, in_range: Optional[Tuple] = None, out_range: Tuple = (0, 1)
) -> ndarray:
    if in_range is None:
        domain_min, domain_max = np.min(x), np.max(x)
    else:
        domain_min, domain_max = in_range
    a, b = out_range
    return a + ((x - domain_min) * (b - a)) / (domain_max - domain_min)
