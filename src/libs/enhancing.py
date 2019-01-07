import math

import numpy as np

import scipy
from scipy import ndimage, signal

import warnings
warnings.simplefilter("ignore")

import cv2
cv2.ocl.setUseOpenCL(False)

from libs.processing import thin_image, clean_points


def enhance_image(image: np.array, block_orientation: int = 16, threshold: float = 0.1,
                  sigma_gradient: int = 1, sigma_block: int = 7, sigma_orientation: int = 7,
                  block_frequency: int = 38, window_size: int = 5, min_wave_length: int = 5,
                  max_wave_length: int = 15, padding: int = None, skeletonise: bool = True):
    """
    Image enhancement using gabor filters based on ridge orientation. 
    Adjusted from https: / / github.com / Utkarsh-Deshmukh / Fingerprint-Enhancement-Python
    Based on the paper: Hong, L., Wan, Y., and Jain, A. K. '
    Fingerprint image enhancement: Algorithm and performance evaluation'. 
    IEEE Transactions on Pattern Analysis and Machine Intelligence 20, 8 (1998), pp 777-789.
    License: BSD 2
    
    """

    # CLAHE adjusted image - histogram equalisation.
    img_clahe = apply_clahe(image)

    # Padding image for applying window frequency mask.
    if padding is not None:
        top, bottom, left, right = [padding] * 4
        img_clahe = cv2.copyMakeBorder(img_clahe, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)

    # Normalise images
    img_normalised, mask = ridge_segment(img_clahe, block_orientation, threshold)

    # Pixel orientation
    img_orientation = ridge_orient(img_normalised, sigma_gradient, sigma_block, sigma_orientation)

    # Ridge frequency
    img_frequency, med = ridge_frequency(img_normalised, mask, img_orientation, block_frequency, window_size,
                                         min_wave_length, max_wave_length)

    # Gabor filter
    image_filtered = ridge_filter(img_normalised, img_orientation, med * mask, .65, .65)

    image_enhanced = (image_filtered < -3)

    if skeletonise:
        # Applies image thinning and sets background to white.
        image_enhanced = thin_image(image_enhanced)
        image_enhanced = clean_points(image_enhanced)

    # Normalising image and processing background - and ridges.
    # image_enhanced = image_enhanced // image_enhanced.max()  # [0, 1] values

    # Invert colours if the background is dark.
    # image_enhanced = swap(image_enhanced) if image_enhanced.mean() < .5 else image_enhanced
    # image_enhanced = image_enhanced.astype('uint8')

    return image_enhanced.astype('uint8')


def ridge_orient(image: np.array, sigma_gradient: int, sigma_block: int, sigma_orientation: int):
    """
    Extracts the orientation of the ridges. 

    """

    # Image gradients.
    size = np.fix(6 * sigma_gradient)

    if np.remainder(size, 2) == 0:
        size = size + 1

    gauss = cv2.getGaussianKernel(np.int(size), sigma_gradient)

    # Gradient of Gaussian
    f = gauss * gauss.T
    fy, fx = np.gradient(f)

    Gx = signal.convolve2d(image, fx, mode='same')
    Gy = signal.convolve2d(image, fy, mode='same')

    Gxx = np.power(Gx, 2)
    Gyy = np.power(Gy, 2)
    Gxy = Gx * Gy

    # Smooth the covariance data to perform a weighted summation of the data.    

    size = np.fix(6 * sigma_block)

    gauss = cv2.getGaussianKernel(np.int(size), sigma_block)
    f = gauss * gauss.T

    Gxx = ndimage.convolve(Gxx, f)
    Gyy = ndimage.convolve(Gyy, f)
    Gxy = 2 * ndimage.convolve(Gxy, f)

    # Analytic solution of principal direction
    denom = np.sqrt(np.power(Gxy, 2) + np.power((Gxx - Gyy), 2)) + np.finfo(float).eps

    # Sine and cosine of doubled angles
    sin2theta = Gxy / denom
    cos2theta = (Gxx - Gyy) / denom

    if sigma_orientation:

        size = np.fix(6 * sigma_orientation)

        if np.remainder(size, 2) == 0:
            size = size + 1

        gauss = cv2.getGaussianKernel(np.int(size), sigma_orientation)

        f = gauss * gauss.T
        cos2theta = ndimage.convolve(cos2theta, f)  # Smoothed sine and cosine of
        sin2theta = ndimage.convolve(sin2theta, f)  # doubled angles

    img_orientation = np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2

    return img_orientation


def ridge_frequency(image: np.array, mask, orient: int, block_size: int, window_size: int, min_wave_length: int,
               max_wave_length: int) -> tuple:
    """
    Ridge frequency computation.

    """

    rows, cols = image.shape
    freq = np.zeros((rows, cols))

    for r in range(0, rows - block_size, block_size):
        for c in range(0, cols - block_size, block_size):
            block_image = image[r: r + block_size][:, c: c + block_size]
            block_orientation = orient[r: r + block_size][:, c: c + block_size]

            freq[r: r + block_size][:, c: c + block_size] = frequest(block_image, block_orientation, window_size,
                                                                     min_wave_length, max_wave_length)

    freq = freq * mask
    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    non_zero_elems_in_freq = freq_1d[0][ind]

    # median_freq = np.mean(non_zero_elems_in_freq)
    # TODO: (Dragos) Review
    median_freq = np.median(non_zero_elems_in_freq)

    return freq, median_freq


def ridge_filter(im, orient, freq, kx, ky):
    angleInc = 3
    im = np.double(im)
    rows, cols = im.shape
    newim = np.zeros((rows, cols))

    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.
    non_zero_elems_in_freq = freq_1d[0][ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

    unfreq = np.unique(non_zero_elems_in_freq)

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.

    sigmax = 1 / unfreq[0] * kx
    sigmay = 1 / unfreq[0] * ky

    sze = np.round(3 * np.max([sigmax, sigmay]))

    x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))

    reffilter = np.exp(-((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay))) * np.cos(
        2 * np.pi * unfreq[0] * x)  # this is the original gabor filter

    filt_rows, filt_cols = reffilter.shape

    gabor_filter = np.array(np.zeros((180 // angleInc, filt_rows, filt_cols)))

    for o in range(0, 180 // angleInc):
        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation  * along *  the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.        

        rot_filt = scipy.ndimage.rotate(reffilter, - (o * angleInc + 90), reshape=False)
        gabor_filter[o] = rot_filt

    # Find indices of matrix points greater than maxsze from the image
    # boundary

    maxsze = int(sze)

    temp = freq > 0
    validr, validc = np.where(temp)

    temp1 = validr > maxsze
    temp2 = validr < rows - maxsze
    temp3 = validc > maxsze
    temp4 = validc < cols - maxsze

    final_temp = temp1 & temp2 & temp3 & temp4

    finalind = np.where(final_temp)

    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees / angleInc)    

    maxorientindex = np.round(180 / angleInc)
    orientindex = np.round(orient / np.pi * 180 / angleInc)

    # do the filtering

    for i in range(0, rows):
        for j in range(0, cols):
            if orientindex[i][j] < 1:
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if orientindex[i][j] > maxorientindex:
                orientindex[i][j] = orientindex[i][j] - maxorientindex

    finalind_rows, finalind_cols = np.shape(finalind)

    sze = int(sze)
    for k in range(0, finalind_cols):
        r = validr[finalind[0][k]]
        c = validc[finalind[0][k]]

        img_block = im[r - sze:r + sze + 1][:, c - sze:c + sze + 1]

        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

    return newim


def normalise(image: np.array):

    normed = (image - np.mean(image)) / (np.std(image))

    return normed


def ridge_segment(im, blksze, thresh):

    rows, cols = im.shape

    im = normalise(im)  # normalise to get zero mean and unit standard deviation

    new_rows = np.int(blksze * np.ceil((np.float(rows)) / (np.float(blksze))))
    new_cols = np.int(blksze * np.ceil((np.float(cols)) / (np.float(blksze))))

    padded_img = np.zeros((new_rows, new_cols))
    stddevim = np.zeros((new_rows, new_cols))

    padded_img[0:rows][:, 0:cols] = im

    for i in range(0, new_rows, blksze):
        for j in range(0, new_cols, blksze):
            block = padded_img[i:i + blksze][:, j:j + blksze]

            stddevim[i:i + blksze][:, j:j + blksze] = np.std(block) * np.ones(block.shape)

    stddevim = stddevim[0:rows][:, 0:cols]

    mask = stddevim > thresh

    mean_val = np.mean(im[mask])

    std_val = np.std(im[mask])

    normim = (im - mean_val) / (std_val)

    return normim, mask


def frequest(im, orientim, windsze, min_wave_length, max_wave_length):
    rows, cols = np.shape(im)

    # Find mean orientation within the block. This is done by averaging the
    # sines and cosines of the doubled angles before reconstructing the
    # angle again.  This avoids wraparound problems at the origin.
    cosorient = np.mean(np.cos(2 * orientim))
    sinorient = np.mean(np.sin(2 * orientim))
    orient = math.atan2(sinorient, cosorient) / 2

    # Rotate the image block so that the ridges are vertical    

    # ROT_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), orient / np.pi * 180 + 90, 1)
    # rotim = cv2.warpAffine(im, ROT_mat, (cols, rows))
    rotim = scipy.ndimage.rotate(im, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode='nearest')

    # Now crop the image so that the rotated image does not contain any
    # invalid regions.  This prevents the projection down the columns
    # from being mucked up.

    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]

    # Sum down the columns to get a projection of the grey values down
    # the ridges.

    proj = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(proj, windsze, structure=np.ones(windsze))

    temp = np.abs(dilation - proj)

    peak_thresh = 2

    maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
    maxind = np.where(maxpts)

    rows_maxind, cols_maxind = np.shape(maxind)

    # Determine the spatial frequency of the ridges by divinding the
    # distance between the 1st and last peaks by the (No of peaks-1). If no
    # peaks are detected, or the wavelength is outside the allowed bounds, 
    # the frequency image is set to 0    

    if cols_maxind < 2:
        freqim = np.zeros(im.shape)
    else:
        peaks = cols_maxind
        wave_length = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (peaks - 1)
        if min_wave_length <= wave_length <= max_wave_length:
            freqim = 1 / np.double(wave_length) * np.ones(im.shape)
        else:
            freqim = np.zeros(im.shape)

    return freqim


def binarise_image(image: np.array, normalise: bool = True) -> np.array:
    """
    OTSU threshold based binarisation
    
    """

    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if normalise:
        # Normalize to 0 and 1 range
        image[image == 255] = 1

    return image


def apply_clahe(image: np.array, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization
    
    """

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    return clahe.apply(image)


def fourier_transform(image: np.array) -> np.array:
    """
    2D Fourier transform image enhancement implementation.
    32x32 pixel windows processed at a time.
    np implementation of FFT for computing DFT
    
    """

    f = np.fft.fft2(image)
    
    return np.fft.fftshift(f)


def high_pass_filter(image: np.array) -> np.array:
    """
    HPF implementation
    
    """

    shifted = fourier_transform(image)
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2
    shifted[crow - 30: crow + 30, ccol - 30: ccol + 30] = 0
    f_ishift = np.fft.ifftshift(shifted)
    
    return np.abs(np.fft.ifft2(f_ishift))
