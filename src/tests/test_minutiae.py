from libs.basics import load_image
from libs.minutiae import process_minutiae
from libs.enhancing import enhance_image

import unittest


class TestMinutiae(unittest.TestCase):
    def test_process_minutiae(self):
        # Tests minutiae creation output.

        path = '../../data/Fingerprints - Set A/101_1.tif'
        img = load_image(path, True)
        img = enhance_image(img, padding=5)

        minutiae = process_minutiae(img)

        self.assertIsNotNone(minutiae, msg='ERROR: Wrong output.')

    def test_plot_minutiae_tree(self):
        pass


if __name__ == '__main__':
    unittest.main()
