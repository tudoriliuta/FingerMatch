# API build
import re
import os
import glob
import time
import operator

import numpy as np

from tqdm import tqdm

from libs.minutiae import plot_minutiae, process_minutiae, minutiae_points, generate_tuple_profile
from libs.edges import match_edge_descriptors
from libs.basics import load_image
from libs.enhancing import image_enhance

from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask.json import jsonify

db_connect = create_engine('sqlite:///test.db')
app = Flask(__name__)
api = Api(app)


class Image:
    """
    Containing element for images - stores image array and its tuple profile.

    """
    def __init__(self, img_id: str, path: str, image: np.array, profile: dict):
        self.img_id = img_id
        self.path = path
        self.image = image
        self.minutiae = None
        self.profile = profile

    def plot(self):
        """
        Plots minutiae from the stored image.

        """

        plot_minutiae(self.image, list(self.tuple_profile.keys()), size=8)


class FingerMatch:
    def __init__(self, model: str = 'tree'):
        self.images = []
        self.model = model

    def loadData(self, path: str, image_format: str = 'tif', limit: int = None) -> None:
        """
        Load data that matches the image_format, from the given path. Each image is processed and stored.

        """

        img_paths = [glob.glob(f'{path}/*.tif', recursive=True)][0]

        try:
            assert len(img_paths) > 0
        except:
            raise FileNotFoundError(f'ERROR: No image files available to extract from the path {path}')

        if limit is not None:
            # Restrict sample size.
            img_paths = img_paths[:limit]

        start = time.time()

        for p in tqdm(img_paths, desc='Loading data.'):
            # Image loading
            image = load_image(p, True)
            image = (image_enhance(image) * 255).astype('uint8')

            try:
                # Image properties definition.
                img_id = re.search(f'(.+?).{image_format}', os.path.basename(p)).group(1)
            except AttributeError:
                raise Exception(f'ERROR: Unknown image id for {p}')

            # Create new profile for the given image and store it.
            self.images.append(Image(img_id, p, image, None))

        print(f'\nINFO: Dataset loaded successfully. Duration: {round(time.time() - start, 2)} sec')

    def trainData(self):
        """
        Loads model on the given dataset.

        """

        start = time.time()
        print('INFO: Loading model features.')

        for i in tqdm(range(len(self.images)), desc='Preparing model.'):
            # Extract minutiae.
            minutiae, _ = process_minutiae(self.images[i].image)

            if self.model == 'tree':
                # Confirmed point matching.
                tuple_profile = generate_tuple_profile(minutiae)
                # Rewriting to the loaded data.
                self.images[i].profile = tuple_profile
                self.images[i].minutiae = minutiae

            elif self.model == 'bf':
                # BFMatcher descriptors generation.
                points, descriptors = minutiae_points(self.images[i].image)
                self.images[i].descriptors = descriptors

        print(f'INFO: Training completed in {round(time.time() - start, 2)} sec')

    def matchFingerprint(self, image: np.array, verbose: bool = False, edge_th: int = 125, match_th: int = 33):
        """
        The given image is compared against the loaded templates.
        A similarity score is computed and used to determine the most likely match, if any.

        """

        if self.model == 'bf':
            # BFMatcher and MES scorings.
            scores = {}

            # Returns score, aims to minimise them for computing the best match.
            # Test descriptors.
            image = (image_enhance(image) * 255).astype('uint8')
            points, descriptors = minutiae_points(image)

            for i in tqdm(range(len(self.images)), desc='Scoring data.'):
                # Matching.
                try:
                    matches = match_edge_descriptors(self.images[i].descriptors, descriptors)
                except AttributeError:
                    raise Exception('ERROR: Model not trained - run trainData first.')

                score = sum([match.distance for match in matches])
                # Using mes (mean edge score) = sum(match distance) / len(matches)
                mes = score / len(matches)

                scores[self.images[i].img_id] = mes

                # Harris edges based feature generation.
                # scores[img.img_id] = match(image_test, self.images[i].image, threshold=None, edge_th=edge_th, visualise=verbose)

            scores = sorted(scores.items(), key=operator.itemgetter(1))

            # Display most likely match
            results = [{'id': i.img_id, 'score': scores[0][1], 'match': scores[0][1] < match_th}
                       for i in self.images if i.img_id == scores[0][0]]

            matches = [m for m in results if m['match']]
            if len(matches) == 0:
                print(f'No match found. Most similar fingerprint is {results[:5]}')
            else:
                print(f'INFO: Matches found, score: {matches}')

            return scores

        elif self.model == 'tree':
            pass

        else:
            print('INFO: Not implemented yet.')



class Images(Resource):
    def get(self):
        conn = db_connect.connect()  # connect to database
        query = conn.execute("select * from Images")  # This line performs query and returns json result
        return {'images': [i[0] for i in query.cursor.fetchall()]}  # Fetches first column that is Employee ID


api.add_resource(Images, '/images')  # Route_1


if __name__ == '__main__':
    app.run(port='5000')
