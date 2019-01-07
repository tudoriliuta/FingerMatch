from libs.basics import load_image
from libs.matching import match_tuples
from libs.minutiae import plot_minutiae, plot_minutiae_tree

import unittest
from main import FingerMatch

PATH = '/Users/Orchestrator/Desktop/Toptal/Fingerprints/dragos-iliuta/data/Fingerprints - Set A'


class TestMain(unittest.TestCase):
    def test_loadData(self):
        # Tests data loading process.
        fm = FingerMatch()

        # with self.assertRaises(FileNotFoundError):
        #    fm.loadData('not_a_path', limit=3)

        fm.loadData(PATH, limit=3)

        self.assertNotEqual(len(fm.images), 0, msg='ERROR: Empty image list. No images were loaded.')

        for i, img in enumerate(fm.images):
            self.assertNotEqual(img.image.size, 0, msg=f'ERROR: Empty image value for the {i}-th image in the class.')

    def test_trainData(self):
        """
        Tests data loading process.

        """

        fm = FingerMatch('bf')
        fm.loadData(PATH, limit=3)

        fm.trainData()

        for i, img in enumerate(fm.images):
            self.assertIsNotNone(img.descriptors, msg=f'ERROR: Profile not built for the {i}-th image in the class.')
            self.assertNotEqual(len(img.descriptors), 0, msg=f'ERROR: Profile not built for the {i}-th image in the class.')

        # Test
        path_test = '../data/Fingerprints - Set B/101_1.tif'

        # Image loading
        img_test = load_image(path_test, True)

        scores = fm.matchFingerprint(img_test, verbose=False)

    def test_matchFingerprintBF(self):
        """

        """

        path_base = '/Users/Orchestrator/Desktop/Toptal/Fingerprints/dragos-iliuta/data/Others/base'
        path_test = '/Users/Orchestrator/Desktop/Toptal/Fingerprints/dragos-iliuta/data/Others/test'

        fm = FingerMatch('bf')

        fm.loadData(path_base, limit=3)
        fm.trainData()

        # Image loading
        img_test = load_image(f'{path_test}/101_2.tif', True)

        fm.matchFingerprint(img_test, verbose=False)

    def test_matchFingerprintTree(self):
        """

        """

        path_base = '/Users/Orchestrator/Desktop/Toptal/Fingerprints/dragos-iliuta/data/Others/base'
        path_test = '/Users/Orchestrator/Desktop/Toptal/Fingerprints/dragos-iliuta/data/Others/test'

        fmt = FingerMatch('tree')
        fmt.loadData(path_base, limit=3)
        fmt.trainData()

        # Image loading
        path_test = f'{path_test}/102_2.tif'
        img_test = load_image(path_test, True)

        fmt.matchFingerprint(img_test, verbose=False)

        # tuple_base = fmt.images[0].profile
        # tuple_test = fmt.images[0].profile
        #
        # confirmed_common_points = match_tuples(tuple_base, tuple_test)

        # Obtains elements sorted from bottom-up and left to right. First element has a lower order.
        # sorted_common_points = sorted(confirmed_common_points, reverse=True)

        # Matching minutiae
        # image = fmt.images[0].image_enhanced
        # minutiae = list(fmt.images[0].profile.keys())
        # plot_minutiae(image, minutiae)

        # plot_minutiae_tree(image, sorted_common_points)


if __name__ == '__main__':
    unittest.main()
