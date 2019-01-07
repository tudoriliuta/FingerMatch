import numpy as np

from skimage.morphology import skeletonize


def clean_points(image: np.array, filter_size: int = 6):
    """
    Remove debris from the image - 1x1 filled pixel areas.

    Args:
        image  (np.array): Image array 1 channel, gray-scale.
        filter_size (int): Size of the rolling window applied for cleaning the array.

    """

    # Normalise
    template_base = image.copy() / image.max()
    
    template_new = template_base.copy()
    
    # Layer - window size.
    height, width = image.shape
    
    for i in range(height - filter_size):
        for j in range(width - filter_size):
            
            layer_filter = template_new[i:i + filter_size, j:j + filter_size]
            
            flag = 0
            if sum(layer_filter[:, 0]) == 0:
                flag += 1
            if sum(layer_filter[:, filter_size - 1]) == 0:
                flag += 1
            if sum(layer_filter[0, :]) == 0:
                flag += 1
            if sum(layer_filter[filter_size - 1, :]) == 0:
                flag += 1
            if flag > 3:
                template_base[i: i + filter_size, j: j + filter_size] = np.zeros((filter_size, filter_size))

    return template_base


def thin_image(image: np.array, swap_image: bool = False) -> np.array:
    """
    Thins given image. 
    
    Args:
        image  (np.array): Image as a numpy array, 1 channel, gray scale.
        swap_image (bool): Swaps background colour with ridge colour
    
    Returns:
        np.array: Thinned image.

    """
    
    # Process image - thinning, cleaning and normalisation
    img_skel = skeletonize(image / image.max()).astype(int)

    # Clean noise or impurities (1x1 filled pixels)
    img_skel = clean_points(img_skel).astype(int)
    
    if swap_image:
        return swap(img_skel)
    else:
        return img_skel


def swap(image: np.array) -> np.array:
    """
    Number swap in np array. Value mirroring.

    Args:
        image (np.array): Image that should be transformed.

    Returns:
        np.array: Transformed image (array)

    """

    image_inverted = np.invert(image)
    
    return image_inverted // image_inverted.max()


def occurrences(image: np.array, max_range: int = 255) -> np.array:
    """
    Probability of occurrence of gray levels in the grayscale image. 
    
    """

    value_distribution = [np.count_nonzero(image == i) for i in range(max_range + 1)]

    return np.array([v / image.size for v in value_distribution])


def cdf(image: np.array) -> np.array:
    """
    Compute np array cdf. 
    
    """

    return np.cumsum(occurrences(image))


def histogram_equalisation(image: np.array) -> np.array:
    """
    Uniformly distributed grayscale data as numpy array. 
    
    """

    mapping = cdf(image) * (occurrences(image) > 0).astype(int)
    scaled_data = mapping * 255

    return np.array([[scaled_data[i] for i in j] for j in image])


def cdf_normalised(image: np.array, max_range: int = 255) -> np.array:
    """
    Normalosed CDF

    """

    hist_data, bins = np.histogram(image.flatten(), max_range, [0, max_range])
    cdf_values = hist_data.cumsum()
    
    return cdf_values * hist_data.max() / cdf_values.max()


def binarise(image: np.ndarray, window_size: int = 8, threshold: int = None, delta: float = .95) -> np.ndarray:
    """    
    Binarise image data - implemented either with or without ridge thinning.
    This function unses a rolling-window of given side size for separating
    the white background from the black ridges based on a threshold or the mean of each frame.
    
    Args:
        image (numpy.ndarray): 1-channel (gray) 2D array storing pixel colour levels.
        window_size     (int): Length of the rolling window's sides. Default: 8
        threshold       (int): Threshold for binarising the values (0 and 255). 
                               If None, it uses the mean value of the frame. Default: None
        delta         (float): Change parameter. Default: .95
    
    Returns:
        Binarised image as a numpy.ndarray
    
    """
    
    # Image dimensions (300x300 as per assignment specifications)
    height = image.shape[0]
    width = image.shape[1]
    
    # Template used to transform the given image. 
    template = image.copy()
        
    for i in range(0, height - window_size + 1, window_size):
        for j in range(0, width - window_size + 1, window_size):
            # Rolling window array and mean of window-contained values.
            window = template[i: i + window_size, j: j + window_size]
            threshold_val = window.mean() if threshold is None else threshold
            
            # Pixel iteration within the window.
            for p in range(0, window.size):
                
                if window[p // window_size][p % window_size] < threshold_val * delta:
                    template[i + p // window_size][j + p % window_size] = 0
                elif window[p // window_size][p % window_size] >= threshold_val * delta:
                    template[i + p // window_size][j + p % window_size] = 255

    # Store as integer to save memory.
    template = template.astype("uint8")
    
    return template
