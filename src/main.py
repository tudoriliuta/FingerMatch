# Class build
import re
import os
import glob
import time
import operator

import numpy as np

from libs.minutiae import plot_minutiae, process_minutiae, generate_tuple_profile
from libs.matching import match_tuples, evaluate
from libs.edges import match_edge_descriptors
from libs.basics import load_image
from libs.enhancing import enhance_image
from libs.edges import edge_processing, sift_match


class Image:
    """
    Containing element for images - stores image array and its tuple profile.

    """
    def __init__(self, img_id: str, path: str, image_raw: np.array, image_enhanced: np.array, profile: dict):
        self.img_id = img_id
        self.path = path
        self.image_raw = image_raw
        self.image_enhanced = image_enhanced
        self.minutiae = None
        self.profile = profile

    def plot(self):
        """
        Plots minutiae from the stored image.

        """

        plot_minutiae(self.image_enhanced, list(self.profile.keys()), size=8)


class FingerMatch:
    def __init__(self, model: str = 'tree', threshold: int = 125):
        self.images = []
        self.model = model
        self.threshold = threshold

    def loadData(self, path: str, image_format: str = 'tif', limit: int = None) -> None:
        """
        Load data that matches the image_format, from the given path. Each image is processed and stored.

        """

        img_paths = [glob.glob(f'{path}/*.{image_format}', recursive=True)][0]

        try:
            assert len(img_paths) > 0
        except:
            raise FileNotFoundError(f'ERROR: No image files available to extract from the path {path}')

        if limit is not None:
            # Restrict sample size.
            img_paths = img_paths[:limit]

        start = time.time()

        for p in img_paths:
            # Image loading
            image_raw = load_image(p, True)

            try:
                # Image properties definition.
                img_id = re.search(f'(.+?).{image_format}', os.path.basename(p)).group(1)
            except AttributeError:
                raise Exception(f'ERROR: Unknown image id for {p}')

            # Create new profile for the given image and store it.
            self.images.append(Image(img_id, p, image_raw, None, None))

        print(f'\nINFO: Dataset loaded successfully. Duration: {round(time.time() - start, 2)} sec')

    def trainData(self):
        """
        Loads model on the given dataset.

        """

        start = time.time()
        print(f'INFO: Loading model features. Model: {self.model.lower()}')

        if self.model.lower() == 'tree':
            for i in range(len(self.images)):
                # Extract minutiae.
                self.images[i].image_enhanced = enhance_image(self.images[i].image_raw, skeletonise=True)
                minutiae = process_minutiae(self.images[i].image_enhanced)

                # Confirmed point matching.
                self.images[i].profile = generate_tuple_profile(minutiae)

                # Rewriting to the loaded data.
                self.images[i].minutiae = minutiae

        elif self.model.lower() == 'orb':
            # Base data.
            print('INFO: Training skipped.')

        elif self.model.lower() == 'bf':
            for i in range(len(self.images)):
                # BFMatcher descriptors generation.
                self.images[i].image_enhanced = enhance_image(self.images[i].image_raw, skeletonise=False)

                points, descriptors = edge_processing(self.images[i].image_enhanced, threshold=self.threshold)

                # points, descriptors = minutiae_points(self.images[i].image_enhanced)
                self.images[i].descriptors = descriptors

        print(f'INFO: Training completed in {round(time.time() - start, 2)} sec')

    def matchFingerprint(self, image: np.array, verbose: bool = False, match_th: int = 33):
        """
        The given image is compared against the loaded templates.
        A similarity score is computed and used to determine the most likely match, if any.

        """

        if self.model.lower() == 'bf':
            # BFMatcher and MES scorings.
            scores = {}

            # Returns score, aims to minimise them for computing the best match.
            # Test descriptors.
            img = enhance_image(image, skeletonise=False)
            points, descriptors = edge_processing(img, threshold=self.threshold)

            for i in range(len(self.images)):
                # Matching.
                try:
                    matches = match_edge_descriptors(self.images[i].descriptors, descriptors)
                except AttributeError:
                    raise Exception('ERROR: Model not trained - run trainData first.')

                # Calculate score
                score = sum([match.distance for match in matches])
                # Using mes (mean edge score) = sum(match distance) / len(matches)
                if len(matches) > 0:
                    mes = score / len(matches)
                    scores[self.images[i].img_id] = mes

            scores = sorted(scores.items(), key=operator.itemgetter(1))

            # Display most likely match
            results = [{'img_id': s[0], 'score': round(s[1], 2), 'match': s[1] < match_th} for s in scores]

            matches = [m for m in results if m['match']]
            if len(matches) == 0:
                print(f'No match found. Most similar fingerprint is {results[:5]}')
            else:
                print(f'INFO: Matches found, score: {matches}')

            return scores

        elif self.model.lower() == 'orb':
            # Basic SIFT based ORB matcher.
            for i in self.images:
                sift_match(i.image_raw, image)

        elif self.model.lower() == 'tree':

            img_test = enhance_image(image, skeletonise=True)

            minutiae_test = process_minutiae(img_test)
            # Confirmed point matching.
            img_profile = generate_tuple_profile(minutiae_test)

            for i in range(len(self.images)):
                # Matching.
                common_points_base, common_points_test = match_tuples(self.images[i].profile, img_profile)

                if evaluate(common_points_base, self.images[i].minutiae, minutiae_test):
                    print(f'Match with {self.images[i].img_id}')
                else:
                    print(f'Not a match with {self.images[i].img_id}')

        elif self.model == 'cnn':
            pass

        else:
            print('INFO: Not implemented yet.')
