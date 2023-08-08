import matplotlib.pyplot as plt
import numpy as np


def imshow_overlay(ax = None, *args):
    '''
    Show multiple images with overlay using matplotlib imshow function.

    Parameters
    ----------
    *args : several tuples
        For each tuple, there several arguments specifing the display parameters.
        
        For the 1st tuple, it should be in the following form:
                (img, [w1,w2], colormap)
            img : an image array
            [w1, w2] : (optional) the window level (can also be set as [] for auto window level)
            colormap : (optional, default is 'gray') colormap for img
            If img has 3 channels (RGB) or 4 channels (RGBA), the rest arguments are ignored.
            
        For the rest tuple, they should be in the following form:
                (color, alpha, [w1, w2])
            color : a list with three number (from 0 to 1) specifying the overlaying color (e.g. [1, 0, 0])
            alpha : the image to be overlayed whose values are as alpha
            [w1, w2] : (optional, default is []) the window to scale 'alpha'

    Returns
    -------
    fig : matplotlib figure
        The whole figure.
    ax : matplotlib axes
        The matplotlib axes.

    '''
    
    # Number of images to be showed
    N = len(args)

    if ax is None:
        fig, ax = plt.subplots()

    #=====================================================================
    # Show the first image (the image layed at the bottom level)
    #=====================================================================
    
    # When the first argument is not a tuple
    if type(args[0]) == tuple:
        arg = args[0]
    else:
        arg = (args[0], )    

    n = len(arg)

    # Show the image
    imgplot = ax.imshow(arg[0], 'gray')

    # Set window level
    if n>=2 and len(arg[1]) == 2:
        imgplot.set_clim(arg[1][0], arg[1][1])

    # Set colormap
    if n == 3:
        imgplot.set_cmap(arg[2])

    #=====================================================================
    # Overlay the rest image
    #=====================================================================
    height, width, *_ = arg[0].shape

    for idx in range(1, N):
        arg = args[idx]
        n = len(arg)

        # Acquire and prepare parameters
        color = arg[0]
        alpha = arg[1]
        img = np.zeros((height, width, 4))

        # Set color
        img[:, :, 0] = color[0]
        img[:, :, 1] = color[1]
        img[:, :, 2] = color[2]
        
        # Scale aplha value
        if n == 3 and len(arg[2]) == 2:
            w1 = arg[2][0]
            w2 = arg[2][1]
        else:
            w1 = np.min(alpha)
            w2 = np.max(alpha)
        alpha = (alpha - w1) / (w2 - w1)

        # Set alpha
        img[:, :, 3] = alpha
        
        # Show the image
        imgplot = ax.imshow(img)
        

    ax.set_xticks([])
    ax.set_yticks([])

    return ax


class IndexTracker:
    def __init__(self, ax, X, **kwargs):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = 0 #self.slices//2

        self.im = ax.imshow(self.X[self.ind], **kwargs)
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        
        
def imshow_3d(img3d: np.ndarray, fig = None, ax = None, **kwargs) -> tuple:
    
    if (fig == None) or (ax == None):
        fig, ax = plt.subplots()
        
    # expand dims if img3d is actually 2d
    if len(img3d.shape) == 2:
        img3d = np.expand_dims(img3d, axis=0)
        
    tracker = IndexTracker(ax, img3d, **kwargs)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    
    return fig, ax, tracker