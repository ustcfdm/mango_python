from numbers import Number
from typing import Sequence

import re
import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F


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
    
    
def extract_substring_between(s: str, sub_left: str, sub_right: str, match_str: str = '.*') -> str:
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
    
    return re.findall(sub_left + f'({match_str})' + sub_right, s)[0]


def rebin_image(img: np.ndarray, bin_size: tuple[int,int], bin_operation: str = 'sum', dtype: str = 'same') -> np.ndarray:
    """Rebin image to make image smaller

    Parameters
    ----------
    img : np.ndarray
        N-dimensional image, where the last two axes are height and width
    bin_size : tuple[int,int]
        Number of bin pixels along row and col direction
    bin_operation : str, optional
        Bin type ('sum' or 'mean'), by default 'sum'
    dype: str, optional
        Output image type, same as img if 'same', by default 'same'

    Returns
    -------
    np.ndarray
        Image after rebin
    
    """
    
    *nslices, nrows, ncols = img.shape
    
    bin_row, bin_col = bin_size
    
    if (nrows % bin_row != 0) or (ncols % bin_col != 0):
        raise ValueError(f'Image shape {nrows,ncols} cannot be divided exactly by bin size {bin_row,bin_col}!')
    
    img = img.reshape((*nslices, nrows//bin_row, bin_row, ncols//bin_col, bin_col))
    
    if dtype == 'same':
        dtype = img.dtype
    
    if bin_operation == 'sum':
        img = np.sum(img, axis=(-3,-1), dtype=dtype)
    elif bin_operation == 'mean':
        img = np.mean(img, axis=(-3,-1), dtype=dtype)
    else:
        raise ValueError(f'Unsupported bin operation "{bin_operation}"!')
    
    return img
    

def median_filter_cuda(img: np.ndarray, size: int | Sequence[int], pad_mode: str = 'constant', value: Number = None, axis: int = 0) -> np.ndarray:
    """Perform median filer for a 2D image using Pytorch CUDA GPU.

    Parameters
    ----------
    img : np.ndarray
        2D image data
    size : int | Sequence[int]
        Median filter size (nrows, ncols)
    pad_mode : str, optional
        'constant', 'reflect', 'replicate', or 'circular', by default 'constant'
    value : Number, optional
        Fill value for 'constant' padding, by default 0
    axis : int, optional
        0 for row by row, 1 for col by col, -1 for all in one (requires large GPU memory), by default 0

    Returns
    -------
    np.ndarray
        2D image after median filtration

    Raises
    ------
    ValueError
        Parameter 'size' must be a int or a sequence of int
    ValueError
        Parameter 'axis' must be one of 0, 1, or -1
    """
    
    # Input image shape
    nrows, ncols = img.shape
    
    # Get filter size
    if isinstance(size, int):
        pad_rows = pad_cols = size
    elif isinstance(size, Sequence):
        pad_rows, pad_cols, *_ = tuple(int(s) for s in size)
    else:
        raise ValueError(f'Invalid type of size: {type(size)}')
    
    # Create Pytorch tensor in GPU
    img_tensor = torch.tensor(img, device='cuda')
    
    # Pad image tensor
    pad = (pad_cols-1)//2, pad_cols//2, (pad_rows-1)//2, pad_rows//2
    img_pad = F.pad(img_tensor, pad, pad_mode, value)
    
    # Peform median filter
    if axis == 0:   # Median filter row by row
        img_med = torch.zeros_like(img_tensor)
        for row in range(nrows):
            img_med[row,:] = img_pad[row:row+pad_rows,:].unfold(0, pad_rows, 1).unfold(1, pad_cols, 1).flatten(-2).median(-1)[0]
    
    elif axis == 1: # Median filter col by col
        img_med = torch.zeros_like(img_tensor)
        for col in range(ncols):
            img_med[:,col] = img_pad[:,col:col+pad_cols].unfold(0, pad_rows, 1).unfold(1, pad_cols, 1).flatten(-2).median(-1)[0].flatten()
            
    elif axis == -1:    # Median filter one stop
        img_med = img_pad.unfold(0, pad_rows, 1).unfold(1, pad_cols, 1).flatten(-2).median(-1)[0]
        
    else:
        raise ValueError(f'Invalid parameter axis={axis}')
    
    return img_med.cpu().numpy()

