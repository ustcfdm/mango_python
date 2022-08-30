import numpy as np

def iqr_limit(a: np.ndarray, percentile: list[float] = [25, 75], ratio: float = 1.5) -> tuple[float, float]:
    
    q1, q3 = np.percentile(a, percentile)
    
    c1 = q1 - ratio * (q3 - q1)
    c2 = q3 + ratio * (q3 - q1)
    
    return c1, c2
