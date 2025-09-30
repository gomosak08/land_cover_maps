import torch
import numpy as np
from path import Path 
from fastai.vision.core import TensorImage
import matplotlib.pyplot as plt





class MSTensorImage(TensorImage):
    """
    The `MSTensorImage` class is a subclass of `TensorImage`, designed for handling multi-spectral tensor images
    with an optional channels-first format. The class includes methods for creating, displaying, and converting
    tensor images.
    
    Attributes:
    - `chnls_first` (bool): Indicates if the tensor image has channels-first format. Default is `False`.
    """

    def __init__(self, x, chnls_first=False):
        """
        Initializes an instance of the `MSTensorImage` class.

        Parameters:
        - `x`: The tensor data to be stored in the instance.
        - `chnls_first` (bool, optional): If `True`, the tensor image is assumed to have channels-first format. Default is `False`.
        """

        self.chnls_first = chnls_first

    @classmethod
    def create(cls, data:(Path), chnls=None, chnls_first=True):
        """
        Class method to create an instance of `MSTensorImage` from a `.npy` file.

        Parameters:
        - `data` (Path): The path to the `.npy` file containing the image data.
        - `chnls` (optional): Channels to be selected from the image data. Default is `None`.
        - `chnls_first` (bool, optional): If `True`, the tensor image is assumed to have channels-first format. Default is `True`.

        Returns:
        - `MSTensorImage`: An instance of the `MSTensorImage` class.
        """
        im = open_npy(fn=data, chnls=chnls, cls=torch.Tensor)
        return cls(im, chnls_first=chnls_first)


    def show(self, chnls=[3, 2, 1], bright=1., ctx=None):
        """
        Displays the tensor image using matplotlib.

        Parameters:
        - `chnls` (list of int, optional): List of channel indices to display. Default is `[3, 2, 1]`.
        - `bright` (float, optional): Brightness adjustment factor. Default is `1.0`.
        - `ctx` (matplotlib.axes.Axes, optional): Matplotlib axes to display the image. If `None`, creates a new plot. Default is `None`.

        Returns:
        - `ctx`: The matplotlib axes used for displaying the image.
        """
        if self.ndim > 2:
            visu_img = self.permute([1, 2, 0])[..., chnls] #self[..., chnls] if not self.chnls_first else self.permute([1, 2, 0])[..., chnls]
        else:
            visu_img = self

        visu_img = visu_img.squeeze()

        visu_img *= bright
        visu_img = np.where(visu_img > 1, 1, visu_img)
        visu_img = np.where(visu_img < 0, 0, visu_img)

        if ctx is None:
            plt.imshow(visu_img)  # Correctly visualize the image
            plt.show()            # Display the plot without arguments
        else:
            ctx.imshow(visu_img)  # Use the provided context for display

        return ctx
    def __repr__(self):
        """
        Provides a string representation of the `MSTensorImage` instance.

        Returns:
        - `str`: A string indicating the shape of the tensor image.
        """
        return (f'MSTensorImage: {self.shape}')
    
    def toNumpy(self, force = True):
        """
        Converts the tensor image to a NumPy array.

        Parameters:
        - `force` (bool, optional): If `True`, forces the conversion to a NumPy array. Default is `True`.

        Returns:
        - `numpy.ndarray`: The converted NumPy array.
        """
        return self.numpy(force = force)




def open_npy(fn, chnls=None, cls=torch.Tensor):
    """
    Opens a .npy file and returns a tensor image.

    Parameters:
    - `fn` (Path): The path to the .npy file.
    - `chnls` (list of int, optional): List of channel indices to select from the image. If `None`, all channels are used. Default is `None`.
    - `cls` (type, optional): The class type to convert the image into. Default is `torch.Tensor`.

    Returns:
    - `cls`: The loaded image as an instance of the specified class.
    """
    im = torch.from_numpy(np.load(str(fn))).to(torch.float32)  # (C,H,W)
    if chnls is not None: im = im[chnls]
    return cls(im)


def open_npy_mask(fn,path, chnls=None, cls=torch.Tensor):

    """
    Opens a .npy file from a specific path and returns a tensor image, specifically for masks.

    Parameters:
    - `fn` (Path): The file name of the .npy file.
    - `path` (str): The directory path where the .npy file is located.
    - `chnls` (list of int, optional): List of channel indices to select from the image. If `None`, all channels are used. Default is `None`.
    - `cls` (type, optional): The class type to convert the image into. Default is `torch.Tensor`.

    Returns:
    - `cls`: The loaded mask image as an instance of the specified class.
    """

    name = path + '/' + fn.name
    im = torch.from_numpy(np.load(str(name))).to(torch.int64)  # (H,W) class indices
    return cls(im)