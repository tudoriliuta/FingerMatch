from itertools import combinations

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from libs.basics import display_image, extract_angle, euclidian_distance


def extract_minutiae(image: np.array):
    """
    Crossing number technique for minutiae extraction from skeletonised binarised images 
    Based on http://airccse.org/journal/ijcseit/papers/2312ijcseit01.pdf
    Requires binarised image array with integer values in [0, 1]. Where 1 is ridge.

    Args:
        image (np.array): Image as a numpy array - 1 channel gray-scale, with white background

    Returns:
        list: [terminations, bifurcations] - extracted from the given image. 
                    terminations (list) - tuple coordinates for the location of a ridge termination
                    bifurcations (list) - tuple coordinates for the location of a ridge bifurcation

    """

    # Index order list - defines the order in which the pixels in a 3x3 frame are considered.
    idx = [(1, -1), (0, -1), (0, 1), (0, 0), (1, 0), (-1, 0), (-1, 1), (-1, -1), (1, -1)]

    debug = False

    height, width = image.shape

    # Store all minutiae
    bifurcations = []
    terminations = []

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # 3x3 frame extraction based on the previous, current and next values on x and y axis.
            frame = image[i - 1: i + 2, j - 1: j + 2]

            # Custom minutiae detection function.
            # Control for pixels found in the middle of the frame.
            # Once identified, it counts filled pixels separated by at least 1 empty pixel.
            pixel_list = [frame[idx[i]] * (1 - frame[idx[i + 1]]) for i in range(len(idx) - 1)]
            pixel_sum = frame[1, 1] * sum(pixel_list)

            # Based on http://airccse.org/journal/ijcseit/papers/2312ijcseit01.pdf
            # pixel_sum = .5 * sum([abs(frame[idx[i]] - frame[idx[i + 1]]) for i in range(len(indices) - 1)])

            if pixel_sum == 1:
                # Termination
                if debug:
                    # Displays a larger frame for debugging purposes.
                    print(f'Termination: {i}, {j}')
                    display_image(image[i - 2: i + 3, j - 2: j + 3])

                # Add termination coordinates
                terminations.append((i, j))

            elif pixel_sum == 3:
                # Bifurcation
                if debug:
                    # Displays a larger frame for debugging purposes.
                    print(f'Bifurcation: {i}, {j}')
                    display_image(image[i - 2: i + 3, j - 2: j + 3])

                # Add bifurcation coordinates
                bifurcations.append((i, j))

    return terminations, bifurcations


def clean_minutiae(image: np.array, minutiae: list) -> list:
    """
    Post-processing
    Remove minutiae identified on the outer terminations of the image.
    We identify outer minutiae as follows: For each type of minutia, we check its quadrant.
    If there are no other full pixels to both the closest sides to an edge on both x and y coord
    That minutiae is discraded.
    Checks location and other pixel values to the sides of the minutiae.
    Outputs list of cleaned minutiae.

    Args:
        image (np.array): Image to be analysed for cleaning borderline minutiae.
        minutiae  (list): Minutiae represented as a list of coordinate tuples (2d: x, y))

    Returns:
        list: Coordinate as tuple list of minutiae that are not found at the image bordering ridge terminations.

    """

    height, width = image.shape

    minutiae_clean = []
    for x, y in minutiae:
        # If there are directions in which the minutiae with x and y coordinates has only empty
        # pixels, that we label the minutiae as an image border and discard it.
        if (image[x, :y].sum() > 0) and (image[x, y + 1:].sum() > 0) and (image[:x, y].sum() > 0) and \
                (image[x + 1:, y].sum() > 0):
            minutiae_clean.append((x, y))

    return minutiae_clean


def extract_tuple_profile(distances: list, m: tuple, minutiae: list) -> list:
    """
    Explores tuple profile. A tuple is a set of minutiae that are found close together.

    Args:
        distances (np.array): Distances between a tuple and its neighbours. Should be used for computing the tuple profile.
        m            (tuple): The base minutiae from which the distances are computed.
        minutiae      (list): List of tuple-like coordinates for all minutiae.

    Returns:
        list: [ratios, angles] - A pair of all angles (list) and all ratios (list) identified for the given tuple.

    """

    # Closest minutiae to the current minutiae
    closest_distances = sorted(distances)[1:6]
    closest_indices = [list(distances).index(d) for d in closest_distances]
    closest_minutiae = [minutiae[i] for i in closest_indices]

    # Unique pair ratios.
    # The 10 pairs used for computing the ratios
    # i-i1 : i-i2, i-i1 : i-i3, i-i1 : i-i4, i-i1 : i-i5,
    # i-i2 : i-i3, i-i2 : i-i4, i-i2 : i-i5
    # i-i3 : i-i4, i-i3 : i-i5
    # i-i4 : i-i5
    unique_pairs = list(combinations(closest_distances, 2))
    # 2 decimal rounded ratios of max of the two distances divided by their minimum.
    compute_ratios = [round(max(p[0], p[1]) / min(p[0], p[1]), 2) for p in unique_pairs]

    # Angle computation.
    minutiae_combinations = list(combinations(closest_minutiae, 2))

    # Angle between the segments drawn from m to the two other minutae with varying distances.
    minutiae_angles = [round(extract_angle((m, x), (m, y)), 2) for x, y in minutiae_combinations]

    return [compute_ratios, minutiae_angles]


def process_minutiae(image: np.array):
    """
    Image processing into minutiae - bifurcations

    Args:
        image   (np.array): Image in 1 channel gray-scale.

    Returns:
        list:     minutiae list containing minutiae coordinates (x, y)

    """

    # Extract minutiae
    terminations, bifurcations = extract_minutiae(image)

    # Post-processing border minutiae removal.
    terminations = clean_minutiae(image, terminations)
    bifurcations = clean_minutiae(image, bifurcations)

    return terminations + bifurcations


def generate_tuple_profile(minutiae: list) -> dict:
    """
    Compute the distance matrix from each minutiae to the rest.

    Args:
        minutiae (list): List of coordinate tuples.

    Returns:
        dict: Tuple profile with all angles and ratios.

    """

    distance_matrix = np.array([[euclidian_distance(i, j) for i in minutiae] for j in minutiae])

    tuples = {}

    for i, m in enumerate(minutiae):
        # When comparing two tuple profiles, one from base and one from test image,
        # they are the same if at least 2 ratios match (and angles).

        # This means that for the tuple profile i is found in a second image under a
        # different tuple's profile.

        # Angles are given a +/- 3.5 degree range to match. To match sourcing device discrepancies.
        ratios_angles = extract_tuple_profile(distance_matrix[i], m, minutiae)
        tuples[m] = np.round(ratios_angles, 2)

    return tuples


def minutiae_points(image: np.array):
    """
    Minutiae as key points.

    """

    # ORB discretises the angle to increments of 2 * pi / 30 (12 degrees) and construct a lookup table of precomputed
    # BRIEF patterns. As long as the keypoint orientationis consistent across views, the correct set of points S
    # will be used to compute its descriptor.
    orb = cv2.ORB_create()

    # Use ORB to detect keypoints.
    points = orb.detect(image)

    # # Use minutiae extracted via crossing numbers technique as keypoints.
    # minutiae = process_minutiae(image)
    # points = [cv2.KeyPoint(y, x, 1) for (x, y) in minutiae]

    # Describe and compute descriptor extractor
    keypoints, descriptors = orb.compute(image, points)

    return keypoints, descriptors


def plot_minutiae_tree(image: np.array, points: list, size: int = 5, node_size: int = 20, graph_color: str = 'blue'):
    """
    Intakes a list of tuple-coordinates that should be linked together via an edge. Plots them on

    image    (np.array): Image array that should be plotted - 1 channel gray-scale
    size          (int): Size of the displayed figure. Square figure with side = size.
    points       (list): List of minutiae coordinates that should be chained together.
    node_size     (int): Graph node size if graph 'G' is given.
    graph_color   (str): Colour of the graph nodes and edges.

    """

    plt.figure(figsize=(size, size))
    plt.imshow(image)
    plt.grid(False)

    G = nx.Graph()

    # Create nodes for each coordinate pair
    for i, coord in enumerate(points):
        G.add_node(i, pos=(coord[1], coord[0]))

    # Create edges between subsequent nodes.
    G.add_edges_from([(i, i + 1) for i in range(len(points[:-1]))])

    nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=False, node_size=node_size, color=graph_color,
            edge_color=graph_color)

    plt.show()


def plot_minutiae(image: np.array, terminations: list = None, bifurcations: list = None, size: int = 5) -> None:
    """
    Plots minutiae as circles on the given image.

    Args:
        image    (np.array): Image array that should be plotted.
        terminations (list): Terminations that should be plotted. Each list element should contain a tuple with the
                                minutiae coordinates.
        bifurcations (list): Bifurcations that should be plotted. Each list element should contain a tuple with the
                                minutiae coordinates.
        size          (int): Size of the displayed figure. Square figure with side = size.

    """

    if bifurcations is None and terminations is None:
        raise Exception("INFO: No 'terminations' or 'bifurcations' parameter given. Nothing to plot.")
    else:
        fig = plt.figure(figsize=(size, size))
        plt.imshow(image)
        plt.grid(False)

    if terminations is not None:
        print("INFO: Plotting terminations\' coordinates")
        for y, x in terminations:
            termination = plt.Circle((x, y), radius=5, linewidth=2, color='red', fill=False)
            fig.add_subplot(111).add_artist(termination)

    if bifurcations is not None:

        print("INFO: Plotting bifurcations\' coordinates")

        for y, x in bifurcations:
            bifurcation = plt.Circle((x, y), radius=5, linewidth=2,
                                     color='blue', fill=False)

            fig.add_subplot(111).add_artist(bifurcation)
