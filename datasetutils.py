# -*- coding: utf-8 -*-
"""\
Utilities for Course 4 - Project.
"""

# Standard library.
import os
import typing as T

# 3rd party.
import numpy as np
import PIL
from PIL import Image  # a module


def image_to_array(path: os.PathLike,
                   rescale: T.Optional[float] = 1/255,
                   target_size: T.Tuple[int, int] = (224, 224),
                   resample: int = Image.BICUBIC ) -> np.ndarray:
    """\
    Loads an image from a file.

    Args:
        path: The path to the file.
        rescale: An optional rescaling factor. If ``None`` or zero, no rescaling is applied. 
            Otherwise, all values are multipled by the rescaling factor.
        target_size: A tuple of integers ``(height, width)`` that specifies the dimensions to which
            the image will be resized.
        resample: The ID of an optional resampling filter for resizing. See
            https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize

    Returns:
        The image, returned as a NumPy array of shape
        ``1 [samples] x height [pixels] x width [pixels] x 3 [color channels]``.
    """
    img = Image.open(path)
    img = img.resize(target_size, resample)
    array = np.asarray(img, dtype=np.float32)  # height x width x 3
    array = array[np.newaxis, :, : :]  # 1 x height x width x 3
    if rescale:  # Do not rescale if None or 0.
        array *= rescale
    return array


def load_images(path: os.PathLike,
                rescale: T.Optional[float] = 1/255,
                target_size: T.Tuple[int, int] = (224, 224),
                resample: int = Image.BICUBIC) -> T.Iterator[T.Tuple[np.ndarray, str]]:
    """\
    Loads images from a directory structure. The names of the sub-directories are interpreted as
    labels for the images that they contain. The expected directory structure is::

        <path>/<label>/<name>.png
  
    Args:
        path: The path to the root of the directory structure.
        rescale: An optional rescaling factor. If ``None`` or zero, no rescaling is applied. 
            Otherwise, all values are multipled by the rescaling factor.
        target_size: A tuple of integers ``(height, width)`` that specifies the dimensions to which
            the image will be resized.
        resample: The ID of an optional resampling filter for resizing. See
            https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize

    Returns:
        An iterator over tuples ``(image, label, name)``. Each image is returned as a NumPy array of
        shape ``1 [samples] x height [pixels] x width [pixels] x 3 [color channels]``.
    """
    label_paths = [entry for entry in path.iterdir() if entry.is_dir()]
    for label_path in sorted(label_paths):
        label = label_path.name
        image_paths = label_path.glob('*.png')
        for image_path in image_paths:
            name = image_path.name
            array = image_to_array(image_path, rescale, target_size, resample)
            yield (array, label, name)


def load_dataset(path: os.PathLike,
                 rescale: T.Optional[float] = 1/255,
                 target_size: T.Tuple[int, int] = (224, 224),
                 resample: int = Image.BICUBIC) -> T.Dict[str, T.Any]:
    """\
    Loads a dataset from a directory structure. The names of the sub-directories are interpreted as
    labels for the images that they contain. The expected directory structure is::

        <path>/<label>/<name>.png

    Args:
        path: The path to the root of the directory structure.
        rescale: An optional rescaling factor. If ``None`` or zero, no rescaling is applied. 
            Otherwise, all values are multipled by the rescaling factor.
        target_size: A tuple of integers ``(height, width)`` that specifies the dimensions to which
            the image will be resized.
        resample: The ID of an optional resampling filter for resizing. See
            https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize

    Returns:
        A dataset as a dictionary with the following entries:
        
        - ``data``: The list of images, as a NumPy array of shape
          ``n [samples] x height [pixels] x width [pixels] x 3 [color channels]``. 
        - ``label_idxs``: The list of numeric labels of the images, as a NumPy array of shape
          ``n [samples]``. Text labels can be reconstructed using ``labels_str[labels_idx]``.
        - ``label_strs``: The list of unique text labels of the images, as a NumPy array of shape
          ``k [categories]``.
        - ``names``: The list of names of the images, as a NumPy array of shape
          ``n [samples]``.
    """
    buf_images = []
    buf_labels = []
    buf_names = []
    for image, label, name in load_images(path, rescale, target_size, resample):
        buf_images.append(image)
        buf_labels.append(label)
        buf_names.append(name)
    # Collect all images into a single array.
    images = np.concatenate(buf_images)
    # Collect all labels into a single array. 
    labels = np.array(buf_labels)
    # Collect all names into a single array.
    names = np.array(buf_names)
    # Figure out numeric indices for the labels.
    label_strs, label_idxs = np.unique(labels, return_inverse=True)
    # We package everything in a dictionary.
    dataset = {
        'data': images,
        'label_idxs': label_idxs,
        'label_strs': label_strs,
        'names': names
    }
    return dataset
