# Minutiae matching for computing similarity score between images. 
import math
import numpy as np
from libs.basics import extract_angle, euclidian_distance, quadrant


class Edge:
    def __init__(self, start: tuple, end: tuple, angle: float, ratio: float, quadrant: int):
        self.start = start
        self.end = end
        self.angle = angle
        self.ratio = ratio
        self.quadrant = quadrant


def match_tuples(tuple_base: dict, tuple_test: dict, th_range: float = .01, th_angle: float = 1.5):
    """
    Comparison between base and test tuples. 
    Test ratios and angles as arrays.
    Minutiae matching for computing similarity score between images. 

    Args:
        tuple_base (dict): Contains the base tuple coordinates as key with a nested list of ratios and angles with
                            the closest 5 neighbours.
        tuple_test (dict): Contains the test tuple coordinates as key with a nested list of ratios and angles with
                            the closest 5 neighbours.
        th_range  (float): Accepted matching threshold for the range criteria. Default: .01
        th_angle  (float): Accepted matching threshold for the angle criteria. Default: 1.5

    Returns:
        list: confirmed matching tuples as a list of coordinates.

    """

    ratios_test = np.array([ratios for c, [ratios, _] in tuple_test.items()])
    angles_test = np.array([angles for c, [_, angles] in tuple_test.items()])

    common_points_base = []
    common_points_test = []

    # Tuple-wise comparison with all tuple profiles in the test image.
    for i, (m, [ratios, angles]) in enumerate(tuple_base.items()):
        # Explore matching ratios.
        matching_values = (ratios_test == ratios).sum(1)

        # Tuples found to match with this base tuple. 
        matching_indices = np.where((matching_values == matching_values.max()) & (matching_values >= 2))[0]

        if len(matching_indices) == 0:
            continue
        else:
            matching_indices = matching_indices[0]

        matching_ratios = ((ratios_test + th_range) >= ratios) * (ratios_test - th_range <= ratios)
        matching_angles = ((angles_test + th_angle) >= angles) * (angles_test - th_angle <= angles)

        matches = ((matching_ratios * ratios_test) * (matching_angles * angles_test) > 0).sum(1)

        if matches.max() >= 2:
            # Ratios and angles belonging to the current tuple are matched with 2 or
            # more ratios and angles from another tuple from the test image. 
            # This is a confirmed common point.
            common_points_base.append(m)
            common_points_test.append(list(tuple_test.keys())[matching_indices])

    return common_points_base, common_points_test


def check_edges(edge_a, edge_b, threshold: float = .05):
    """
    Check angle match

    """
    angle_match = math.isclose(edge_a.angle, edge_b.angle, rel_tol=threshold)
    ratio_match = math.isclose(edge_a.ratio, edge_b.ratio, rel_tol=threshold)
    quadrant_match = math.isclose(edge_a.quadrant, edge_b.quadrant, rel_tol=threshold)

    is_identical = angle_match and ratio_match and quadrant_match

    return is_identical


def edge_matching(edges_base: list, edges_test: list):

    matched_edges = []

    for i in range(max(len(edges_base), len(edges_test))):
        if check_edges(edges_base[i], edges_test[i]):
            # Common edge that matches test image.
            matched_edges.append(edges_base[i])
        else:
            # Check both the image test +1 or image base +1
            if check_edges(edges_base[i], edges_test[i + 1]):
                # Remove the current edges_test node.
                pass
            elif check_edges(edges_base[i + 1], edges_test[i]):
                # Remove the current edges_base node.
                pass


def evaluate(common_points, minutiae_base, minutiae_test):
    """
    Currently using the following metric:
    Common points >= max(total minutiae across both images) / 2

    """

    minutiae_score = max(len(minutiae_base), len(minutiae_test)) / 2

    print(f'INFO: Score - {len(common_points) / minutiae_score}')

    return len(common_points) >= minutiae_score


def build_edges(common_points, sample_size: int = None, verbose: bool = False):
    """
    Generate edge tree.

    """

    edges = []

    sample_size = len(common_points) if sample_size is None else sample_size

    for i in range(1, sample_size - 1):

        # 300 - y, 300 - x
        seg_a = [common_points[i - 1], common_points[i]]
        seg_b = [common_points[i], common_points[i + 1]]

        angle = extract_angle(seg_a, seg_b)
        Q = quadrant(*seg_b)

        seg_a_length = euclidian_distance(*seg_a)
        seg_b_length = euclidian_distance(*seg_b)

        # Segment length ratio
        a_b_ratio = max(seg_a_length, seg_b_length) / min(seg_a_length, seg_b_length)

        if verbose:
            print(f'INFO: Edge #{i} Quadrant: {direction} AB Ratio: {round(a_b_ratio, 2)} Angle: {round(angle, 2)}',
                  seg_a, seg_b)

        edges.append(Edge(seg_b[0], seg_b[1], angle, a_b_ratio, Q))

    return edges


def edge_matching(edges_base: list, edges_test: list):
    """
    Tree matching between base and test image profiles.

    """

    pass
