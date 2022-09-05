import numpy as np

def iqr_limit(a: np.ndarray, percentile: list[float] = [25, 75], ratio: float = 1.5) -> tuple[float, float]:
    """Calculate outlier limit values using IQR method.

    Parameters
    ----------
    a : np.ndarray
        Data array
    percentile : list[float], optional
        Percentile value, by default [25, 75]
    ratio : float, optional
        Range ratio, by default 1.5

    Returns
    -------
    tuple[float, float]
        Lower outlier limit, and upper outlier limit
    """
    
    q1, q3 = np.percentile(a, percentile)
    
    c1 = q1 - ratio * (q3 - q1)
    c2 = q3 + ratio * (q3 - q1)
    
    return c1, c2


def get_window_level_iqr(img: np.ndarray, percentile: list[float] = [25, 75], ratio: float = 1.5) -> tuple[float, float]:
    """Get window level for an image using IQR method.

    Parameters
    ----------
    img : np.ndarray
        Image array
    percentile : list[float], optional
        Percentile, by default [25, 75]
    ratio : float, optional
        Range ratio, by default 1.5

    Returns
    -------
    tuple[float, float]
        Window level
    """
    
    c1, c2 = iqr_limit(img, percentile, ratio)
    
    w1 = max(np.min(img), c1)
    w2 = min(np.max(img), c2)
    
    return w1, w2
