import re
import numpy as np
from scipy import signal


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


def index_outliers(a: np.ndarray, percentile: list[float] = [25, 75], ratio: float = 1.5) -> np.ndarray:
    """Calculate outlier limit and find out index of outliers.

    Parameters
    ----------
    a : np.ndarray
        N-dimensional array
    percentile : list[float], optional
        Percentile, by default [25, 75]
    ratio : float, optional
        Range ratio, by default 1.5

    Returns
    -------
    np.ndarray
        Bool type, indicating outlier elements
    """
    
    # get IQR limit
    c1, c2 = iqr_limit(a, percentile, ratio)
    
    # index of outliers
    a_outlier = (a < c1) | (a > c2)
    
    return a_outlier


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


def fix_bad_points(a: np.ndarray, median_size: int = 5, iqr_paras: list[float] = [25, 75, 1.5]) -> np.ndarray:
    # TODO: add comment
    # a: n-dimensional array
    
    # apply median filter
    a_smooth = signal.medfilt(a, kernel_size=median_size).astype('float')
    
    # noise
    a_noise = a - a_smooth
    
    # find out bad points
    idx_bp = index_outliers(a_noise, [10, 90])
    
    # replace bad points with median value
    a[idx_bp] = a_smooth[idx_bp]
    
    return a
    
    
def extract_substring_between(s: str, sub_left: str, sub_right: str) -> str:
    """Extract substring between two substrings from a string.

    Parameters
    ----------
    s : str
        Input string
    sub_left : str
        Left substring
    sub_right : str
        Right substring

    Returns
    -------
    str
        Output substring
    """
    
    return re.findall(sub_left + '(.*)' + sub_right, s)[0]