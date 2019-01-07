from libs.basics import load_image, display_image
from libs.enhancing import enhance_image

import unittest


class TestEnhancing(unittest.TestCase):
    def test_enhance_image(self):
        # Tests image enhancing output and data loss.

        # Revert gray colour levels. Match scale with the raw image for comparison.
        path = '../../data/Fingerprints - Set A/101_1.tif'

        # Image loading
        img = load_image(path, True)

        img = enhance_image(img, padding=5)


if __name__ == '__main__':
    unittest.main()
