import cv2
import math

import numpy as np

from PIL import Image

from sklearn.decomposition import PCA
from scipy.spatial import distance

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
# matplotlib.use("MacOSX")


def load_image(path: str, gray: bool = False) -> np.array:
    """
    Loads an m*n dimensional numpy array

    """

    if gray:
        return cv2.imread(path, 0)
    else:
        return cv2.imread(path)


def get_angle(a, b):
    """
    Computes the angle of a segment AB with A(a[1], a[0]) and B(b[1], b[0])

    """

    # Absolute value - increasing order already established.
    radians = math.atan2(abs(b[0] - a[0]), b[1] - a[1])
    return np.rad2deg(radians)


def quadrant(a, b):
    """
    Returns the quadrant of the given angle degrees.

    """

    return int(get_angle(a, b) // 90 + 1)


def euclidian_distance(m1: tuple, m2: tuple) -> float:
    """
    Distance between 2 points based on their 2D coordinates

    Args:
        m1 (tuple): Coordinates (x, y) used as the first distance measurement point.
        m2 (tuple): Coordinates (x, y) used as the second distance measurement point.

    Returns:
        int: Distance between the two coordinates using euclidian distance (Pythagorean theorem)

    """

    return distance.euclidean(m1, m2)


def extract_angle(a: list, b: list) -> float:
    """
    Extract angle between two vectors with endpoints defined by two tuples.

    Args:
        a            (list): First segment that contains a starting coordinate (x, y) and an ending coordinate (x, y)
        b            (list): Second segment that contains a starting coordinate (x, y) and an ending coordinate (x, y)
        centre_angle (bool): True - free angle, False - constrained in the range [0, 180]

    Returns:
        float: Angle between the two segments.

    """

    # Vector form
    a_vec = [(a[0][0] - a[1][0]), (a[0][1] - a[1][1])]
    b_vec = [(b[0][0] - b[1][0]), (b[0][1] - b[1][1])]

    ab_dot = np.dot(a_vec, b_vec)
    a_mag = np.dot(a_vec, a_vec) ** 0.5
    b_mag = np.dot(b_vec, b_vec) ** 0.5

    # Radian angle to degrees
    angle = math.acos(round(ab_dot / b_mag / a_mag, 6))

    angle_degrees = math.degrees(angle) % 360

    if angle_degrees - 180 >= 0:
        return 360 - angle_degrees
    else:
        return angle_degrees


def image_resize(image: np.array, size: tuple):
    """
    Resizes given image object.

    Args:
        image (nd.array): Image array which should be resized.
        size     (tuple): Structure: (width(float), height(float) - shape of the resized image.

    Returns:
        nd.array: Resized image.

    """

    return cv2.resize(image, size)


def save_image(image: np.array, path: str):
    """
    Writes image to path.

    Args:
        image (nd.array): Image array that should be saved.
        path       (str): Save path.

    """

    cv2.imwrite(image, path)


def display_image(image: np.array, title: str = None, cmap: str = None, figsize: tuple = None):
    """
    Plots an image using matplotlib pyplot imshow.

    Args:
        image (nd.array): Image that should be visualised.
        title      (str): Displayed graph title.
        cmap       (str): Cmap type.
        figsize  (tuple): Size of the displayed figure. 

    """

    if figsize:
        plt.figure(figsize=figsize)

    plt.imshow(image, cmap=cmap)

    if (len(image.shape) == 2) or (image.shape[-1] == 1):
        plt.gray()

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(title)
    plt.show()


def display_dataset(dataset, root_path: str = '../data', users=10, samples=8, size=2):
    """
    Function to return dataset fingerprints. 
    
    """

    images = [load_image(f'{root_path}/Fingerprints - Set {dataset}/10{i+1}_{j+1}.tif', True) for i in range(users) for j in range(samples)]

    for i in range(users):
        plt.figure(figsize=(size * 10, size))
        display_image(np.hstack(images[i:i+8]))


def array_to_image(image):
    """
    Returns a PIL Image object

    """

    return Image.fromarray(image)


def gaussian_filter(image, shape=(5, 5), dx=0, dy=0):
    """
    Applies a Gaussian smoothing filter to an image array.

    Args:
        image (nd.array): Image array which should be smoothed.
        shape    (float): Odd and positive numbers - size of the kernel.
        dx       (float): Std of x.
        dy       (float): Std of y.

    Returns:
        nd.array: Gaussian filter on top of an image numpy array.

    """

    return cv2.GaussianBlur(image, shape, dx, dy)


def laplace_filter(image, ddepth=cv2.CV_64F, ksize=1):
    """
    Applies Laplace filter to an image array.
    This will identify areas of rapid change.

    Args:
        image (nd.array): Image array which should be transformed.
        ksize      (int): Laplace filter type. s

    Returns:
        nd.array: Transformed image.

    """

    return cv2.Laplacian(image, ddepth=ddepth, ksize=ksize)


def laplacian_of_gaussian(image_gray, sigma=1., kappa=0.75, pad=False):
    """
    Applies Laplacian of Gaussians to grayscale image.
    # https://github.com/amirdel/stanfordHacks/tree/master/notebooks

    Args:
        image_gray (nd.array): Image to apply the laplacian of Gaussian on top of.
        sigma:        (float): Sigma of Gaussian applied to image. <= 0 is None
        kappa:        (float): Difference threshold as factor to mean of image values, <= 0 is None
        pad:          (bool): Flag to pad output w/ zero border, keeping input image size

    Returns:
        nd.array: Transformed image.
    """

    assert len(image_gray.shape) == 2
    image = cv2.GaussianBlur(image_gray, (0, 0), sigma) if 0 < sigma else image_gray
    image = cv2.Laplacian(image, cv2.CV_64F)
    rows, cols = image.shape[:2]

    # Min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(image[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(image[r:rows - 2 + r, c: cols - 2 + c]
                                     for r in range(3) for c in range(3)))

    # Bool matrix for image value positiv (w/out border pixels)
    image_pos = 0 < image[1:rows - 1, 1:cols - 1]

    # Bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - image_pos] = 0

    # Bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[image_pos] = 0

    # Sign change at pixel
    zero_cross = neg_min + pos_max

    # Values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., image.max() - image.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.

    # Optional Thresholding
    if 0. <= kappa:
        thresh = float(np.absolute(image).mean()) * kappa
        values[values < thresh] = 0.

    image_log = values.astype(np.uint8)

    if pad:
        image_log = np.pad(image_log, pad_width=1, mode='constant', constant_values=0)

    return image_log


def laplacian_variance(image):
    """
    Applies a Laplace filter on an image and returns its variance.

    """

    return cv2.Laplacian(image, cv2.CV64F).var()


def slide_template(image, wsize: int):
    """
    Builds a sliding template.

    Args:
        image (nd.array): Image array.
        wsize      (int): Step size.

    Returns:
        nd.array: Enhanced image.
    """

    new_image = []
    for col in range(image.shape[1] - wsize - 1):
        for row in range(image.shape[0] - wsize - 1):
            window = []
            for k in range(wsize):
                for l in range(wsize):
                    window.append(image[row + k, col + l])
            new_image.append(window)
    return np.asarray(new_image)


def pca(image):
    """
    Perform principal component analysis (PCA) on the given image.

    Args:
        image (nd.array): Image as array onto which PCA is performed.

    Returns:
        float:

    """

    pca = PCA(n_components=3)
    scores = pca.fit_transform(image)
    return scores, pca.explained_variance_ratio_


def grayscale(image):
    """
    Using Rec.ITU-R BT.601-7 - conversion from RGB to greyscale

    """

    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def image_to_array(image: np.array, mode: str = 'LA'):
    """
    Returns an image 2D array.

    """

    image_array = np.fromiter(iter(image.getdata()), np.uint8)
    image_array.resize(image.height, image.width)

    return image_array
