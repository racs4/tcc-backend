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

if __name__ == '__main__':
    seed(1)
    # ellipse = {
    #     "center": [
    #         40.685814225322915,
    #         0.5309239579620285
    #     ],
    #     "height": 97.64819874927565,
    #     "phi": 0.02034893891436118,
    #     "width": 17.70021263833929
    # }
    # luminances = {
    #     "top": 55.859375,
    #     "bottom": 37.890625
    # }
    # confusion_point = [
    #     257.04166062615747,
    #     4.933535860137521
    # ]

    ellipse = {
        "center": [
          -73.77114992493914,
          6.106258598586408
        ],
        "height": 15.735585987622771,
        "phi": -0.058370397376381784,
        "width": 91.66005948620332
    }
    luminances = {
        "top": 55.859375,
        "bottom": 46.484375
    }
    confusion_point = [
        -25.830984638903384,
        826.482842779623
    ]
   
    image_names = ["flor", "cidade", "floresta", "ishihara", "maca"]
    image_dir = pathlib.Path('images')
    image_dir_save = pathlib.Path('images2')

    for image_name in image_names:
        image = Image.open(image_dir.joinpath(
            f'{image_name}-original.png')).convert('RGB')
        
        # creates the kmeans model used in both
        n_clusters = int(log(image.size[0] * image.size[1]))
        model_tcc = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=1)

         # creates the tcc recolor algorithm
        recolor_algorithm_tcc = Recolor(
            Cluster(model_tcc, resize=False, maxColors=20000),
            GraphGenerator(ellipse, confusion_point, luminances),
            GeneticRecolor(ellipse, confusion_point, luminances, random_state=1),
            RecolorCluster(model_tcc)
        )
        
        # save image with tcc colors
        (new_image, color_dict) = recolor_algorithm_tcc.execute(image)
        new_image = Image.fromarray(new_image)
        new_image.save(image_dir_save.joinpath(f'{image_name}-recolored-tcc.png'))

        # creates the original recolor algorithm
        recolor_algorithm_original = Recolor(
            Cluster(model_tcc, resize=False, maxColors=20000),
            GraphGenerator(ellipse, confusion_point, luminances),
            RandomRecolor(ellipse, confusion_point, luminances, random_state=1),
            RecolorCluster(model_tcc)
        )

        # save image with original recolor colors
        (new_image, color_dict) = recolor_algorithm_original.execute(image)
        new_image = Image.fromarray(new_image)
        new_image.save(image_dir_save.joinpath(f'{image_name}-recolored-original.png'))

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # for (key, value) in color_dict.items():  # type: ignore
    #     ax.plot(key[1], key[2], key[0], 'bo')
    #     if (key != value):
    #         ax.plot(value[1], value[2], value[0], 'ro')
    #         ax.plot([value[1], key[1]], [value[2], key[2]], [value[0], key[0]])

    # plt.show()
