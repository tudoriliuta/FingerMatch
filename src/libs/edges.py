import numpy as np

from libs.basics import display_image
from libs.enhancing import enhance_image
from libs.processing import thin_image, swap

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
# matplotlib.use("MacOSX")

import cv2
cv2.ocl.setUseOpenCL(False)


def harris_edges(image: np.array, threshold: int = 200, block_size: int = 3, aperture_size: int = 3, k: float = .04) -> list:
    """
    Runs the Harris edge detector on the image. 
    For each pixel (x, y) it calculates a 2 x 2 gradient covariance matrix M(x,y) over a 
    block_size x block_size neighborhood. 

    Computes:
        dst(x, y) = det M(x, y) - k * (tr M(x, y))^2

    Args:
        image    (np.array): Single-channel image. Shape (x, y)
        threshold     (int): Harris edge threshold. Default: 200
        block_size    (int): Harris corners block size. Default: 3
        aperture_size (int): Aperture in harris edge computation. Default: 3
        k           (float): Edge computation scaling parameter. Default: .04

    """
    # Harris edges
    corners = cv2.cornerHarris(image, block_size, aperture_size, k)
    normalised = cv2.normalize(corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

    # Extract minutiae
    # print(f'INFO: Computing Harris edges using a threshold = {threshold}')
    edges = []
    for x in range(normalised.shape[0]):
        for y in range(normalised.shape[1]):
            if normalised[x][y] > threshold:
                edges.append(cv2.KeyPoint(y, x, 1))

    return edges



def edge_processing(image: np.array, threshold: int = 200):
    """
    Extract Harris edges on the enhanced image.

    """

    # keypoints = harris_edges(image, threshold=threshold)

    # Describe and compute descriptor extractor
    orb = cv2.ORB_create()

    keypoints, descriptor = orb.detectAndCompute(image, None)
    
    # descriptor = orb.compute(image, keypoints)[1]

    return keypoints, descriptor


def plot_edges(image_base: np.array, image_test: np.array, edges_base: list, edges_test: list):
    """
    Plot Harris edges against the base and test images and their matches.

    """

    # Plot keypoints
    img_base = cv2.drawKeypoints(image_base, edges_base, outImage=None)
    img_test = cv2.drawKeypoints(image_test, edges_test, outImage=None)

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img_base)
    ax[0].grid(False)
    ax[1].imshow(img_test)
    ax[1].grid(False)
    plt.show()


def plot_matches(image_base: np.array, image_test: np.array, edges_base: list, edges_test: list, matches: list):
    """
    Plot identified matches.

    """

    img_transform = cv2.drawMatches(image_base, edges_base, image_test, edges_test, matches, flags=2, outImg=None)
    plt.imshow(img_transform)
    plt.grid(False)
    plt.show()


def match_edge_descriptors(descriptor_base: np.array, descriptor_test: np.array, match_function=cv2.NORM_HAMMING) -> list:
    """
    Edge, descriptor matching. Using brute force match constructor. 

    Args:
        descriptor_base (np.array): Base descriptor array
        descriptor_test (np.array): Test descriptor array

    Returns:
        list: matches

    """

    bf = cv2.BFMatcher(match_function, crossCheck=True)

    # Distance based matching
    matches = sorted(bf.match(descriptor_base, descriptor_test), key=lambda match: match.distance)

    return matches


def sift_match(image_base: np.array, image_test: np.array):

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_base, None)
    kp2, des2 = sift.detectAndCompute(image_test, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    match_mask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            match_mask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=match_mask,
                       flags=0)

    img3 = cv2.drawMatchesKnn(image_base, kp1, image_test, kp2, matches, None, **draw_params)

    plt.imshow(img3, )
    plt.show()


def match(image_base: np.array, image_test: np.array, threshold: int = None, edge_th: int = 200, visualise: bool = False):
    """ 
    Full edge matching

    """

    # Edges and descriptors.
    edges_base, descriptors_base = edge_processing(image_base, edge_th)
    edges_test, descriptors_test = edge_processing(image_test, edge_th)

    # Matching.
    matches = match_edge_descriptors(descriptors_base, descriptors_test)
    score = sum([match.distance for match in matches])
    
    # Using mes (mean edge score) = sum(match distance) / len(matches)
    # Based on 
    mes = score / len(matches)

    # Plotting
    if visualise:
        plot_edges(image_base, image_test, edges_base, edges_test)
        plot_matches(image_base, image_test, edges_base, edges_test, matches)

    if threshold is not None:
        # print(f'INFO: Threshold is set to {threshold} - returning match classification. MES={mes}')

        # Classify match
        if mes < threshold:
            # Match identified
            return True
        else:
            # Not a match
            return False
    else:
        # print('INFO: Threshold is set to None - returning match score')
        return mes
