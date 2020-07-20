from PIL import Image
import numpy as np

def read_tiff(filename : str) -> np.ndarray:
    """
    filename - file name of the multipage-tiff 
    """
    img = Image.open(filename)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)