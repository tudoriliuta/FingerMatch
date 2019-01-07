from libs.basics import load_image, display_image
from libs.minutiae import process_minutiae, generate_tuple_profile
from libs.enhancing import enhance_image
from libs.matching import match_tuples, build_edges, edge_matching

import unittest

path_base = '../../data/Fingerprints - Set A/101_2.tif'
path_test = '../../data/Fingerprints - Set A/101_2.tif'

img_base = load_image(path_base, True)
img_base = enhance_image(img_base, padding=5)

img_test = load_image(path_test, True)
img_test = enhance_image(img_test, padding=5)

# Confirmed point matching.
TUPLE_BASE = generate_tuple_profile(process_minutiae(img_base))
TUPLE_TEST = generate_tuple_profile(process_minutiae(img_test))


class TestMatching(unittest.TestCase):
    def test_match_tuples(self):
        # Tests minutiae creation output.

        ccpb_base, ccpb_test = match_tuples(TUPLE_BASE, TUPLE_TEST)

        print('---')

    def test_edge_matching(self):

        common_points_base, common_points_test = match_tuples(TUPLE_BASE, TUPLE_TEST)
        sorted_common_points_base = sorted(common_points_base, reverse=True, key=lambda x: x[0])
        sorted_common_points_test = sorted(common_points_test, reverse=True, key=lambda x: x[0])

        edges_base = build_edges(sorted_common_points_base)
        edges_test = build_edges(sorted_common_points_test)

        edge_matching(edges_base, edges_test)


if __name__ == '__main__':
    unittest.main()
