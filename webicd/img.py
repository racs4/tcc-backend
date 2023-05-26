from math import log
import pathlib
from random import seed
from tkinter import image_names
from webicd.recolorConfusionColorStrategies.genetic import GeneticRecolor
from webicd.recolorStrategies.cluster import RecolorCluster
from webicd.recolorConfusionColorStrategies.random import RandomRecolor
from webicd.findConfusionColorStrategies.graph import GraphGenerator
from webicd.recolor import Recolor
from webicd.getColorStrategies.cluster import Cluster
from PIL import Image
import sklearn.cluster
import matplotlib.pyplot as plt


def recolor_image(image_name, algorithm, ellipse, confusion_point, luminances):
    # print(f'Recoloring image {image_name} with algorithm {algorithm}')
    image_dir = pathlib.Path('images')

    image = Image.open(image_dir.joinpath(
        f'{image_name}-original.png')).convert('RGB')

    # creates the kmeans model used in both
    n_clusters = int(log(image.size[0] * image.size[1]))
    model_tcc = sklearn.cluster.KMeans(
        n_clusters=n_clusters, random_state=1)

    if algorithm == 'tcc':
        # creates the tcc recolor algorithm
        recolor_algorithm_tcc = Recolor(
            Cluster(model_tcc, resize=False, maxColors=20000),
            GraphGenerator(ellipse, confusion_point, luminances),
            GeneticRecolor(ellipse, confusion_point,
                           luminances, random_state=1),
            RecolorCluster(model_tcc)
        )

        # save image with tcc colors
        (tcc_image, color_dict) = recolor_algorithm_tcc.execute(image)
        tcc_image = Image.fromarray(tcc_image)
        # tcc_image_bytes = tcc_image.tobytes()
        return tcc_image
    else:
        # creates the original recolor algorithm
        recolor_algorithm_original = Recolor(
            Cluster(model_tcc, resize=False, maxColors=20000),
            GraphGenerator(ellipse, confusion_point, luminances),
            RandomRecolor(ellipse, confusion_point,
                          luminances, random_state=1),
            RecolorCluster(model_tcc)
        )

        # save image with original recolor colors
        (original_image, color_dict) = recolor_algorithm_original.execute(image)
        original_image = Image.fromarray(original_image)
        # original_image = original_image.tobytes()
        return original_image
