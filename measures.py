from argparse import ArgumentParser
from math import log, log2, sqrt
import pathlib
from typing import Tuple
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans, Birch
from heatmap import annotate_heatmap, heatmap
from webicd.icd import differentiation
from webicd.utils import color_distance, get_colors
from webicd.convert import rgb_to_luv_gama, luv_gama_to_rgb
from scipy.stats import kruskal
import scikit_posthocs as sp

image_tags = ["original", "deuteranopia", "protanopia",
              "tritanopia", "recolored-tcc", "recolored-original"]

image_labels = ["Original", "Deuteranopia",
                "Protanopia", "Tritanopia", "Algoritmo novo", "Algoritmo antigo"]


def dist_img(img_path1: pathlib.Path, img_path2: pathlib.Path):
    """
    Calculates the difference between two images.
    """
    # print(f'Calculating distance between {img_path1} and {img_path2}')

    # load images
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')

    # convert to LUV
    colors1 = np.apply_along_axis(
        rgb_to_luv_gama, 2, np.asarray(img1))
    colors2 = np.apply_along_axis(
        rgb_to_luv_gama, 2, np.asarray(img2))

    # calculate difference
    dist_func = np.vectorize(lambda colors: color_distance(
        colors[0], colors[1]), signature='(m, n)->()')
    dists = dist_func(np.stack([colors1, colors2], axis=2))

    return np.mean(dists), np.median(dists), np.std(dists), dists.reshape(-1)


def diff_img(img_path: pathlib.Path, ellipse, confusion_point, luminances):
    """
    Calculates the differentiation between the points of the image.
    """

    # print(f'Calculating differentiation among {img_path}')

    # load image
    img = Image.open(img_path).convert('RGB')
    # colors = np.array(img.getpalette(),dtype=np.uint8).reshape((256,3))

    # convert to LUV
    colors = np.array(img)
    colors = np.apply_along_axis(
        rgb_to_luv_gama, 2, colors)
    colors = colors.reshape(np.product(
        colors.shape[:2]), colors.shape[2]).astype(float)

    n_clusters = int(log(colors.size / 3))
    # print(f'Clustering {colors.size} points into {n_clusters} clusters')
    key_colors = MiniBatchKMeans(n_clusters=n_clusters, random_state=1).fit(
        colors).cluster_centers_

    # key_colors = colors

    gen_pallete(key_colors, name=img_path)

    diffs = []
    # calculate differentiation
    for i in range(len(key_colors)):
        colorA = key_colors[i]
        for j in range(i+1, len(key_colors)):
            colorB = key_colors[j]
            diff = differentiation(
                ellipse, confusion_point, luminances, colorA, colorB)
            # print(f'{colorA} - {colorB} = {diff}')
            diffs.append(diff)

    return np.mean(diffs), np.median(diffs), np.std(diffs), diffs


def gen_pallete(key_colors, name: pathlib.Path):
    """
    Generates a palette from the key colors.
    """
    # file name
    file = name.parts[-1]

    # reshape to (m, n, 3)
    key_colors = key_colors.reshape(
        key_colors.shape[0], 1, key_colors.shape[1])

    key_colors = np.sort(key_colors, axis=0)

    # convert to RGB
    key_colors = np.apply_along_axis(
        luv_gama_to_rgb, 2, key_colors).astype(np.uint8)

    key_colors = key_colors.repeat(50, axis=1)
    key_colors = key_colors.repeat(50, axis=0)

    Image.fromarray(key_colors).rotate(
        90, expand=True).save(pathlib.Path("palletes2").joinpath(file))


def calculate_statistical_test(name, measure, groups_arr):
    k_result = kruskal(*groups_arr)
    # print(k_result)
    n_result = sp.posthoc_nemenyi(np.array(groups_arr), sort=True)
    # print(n_result)

    fig, ax = plt.subplots()
    ax.set_title(f'Valor p-value de Nemenyi de {measure} para "{name}"')
    im, cbar = heatmap(n_result, image_labels, image_labels, ax=ax,
                       cmap="YlGn_r", vmin=0, vmax=1, cbarlabel="p-values")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    fig.tight_layout()
    plt.savefig(pathlib.Path("measures").joinpath(
        f'heatmap-{measure}-{name}.png'))

    fig7, ax7 = plt.subplots()
    ax7.set_title('Boxplot das amostras de {} para "{}"'.format(measure, name))
    ax7.boxplot(groups_arr, showfliers=False, labels=image_labels)
    plt.xticks(rotation = 90)
    fig7.tight_layout()
    plt.savefig(pathlib.Path("measures").joinpath(
        f'boxplot-{measure}-{name}.png'))


plt.show()

# testing
if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        '-t', '--tags', metavar='str', type=str, choices=image_tags,
        nargs='+', default=image_tags)
    args = parser.parse_args()

    # user discrimination
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

    image_tags = args.tags
    image_names = ["flor", "cidade", "floresta", "ishihara", "maca"]
    image_dir = pathlib.Path('images2')

    for name in image_names:
        dists_arr = []
        diffs_arr = []
        # print(f'{name}')
        for tag in image_tags:
            # print(f'\t{tag}')
            (diff_mean, diff_median, diff_std, diffs) = diff_img(image_dir.joinpath(f'{name}-{tag}.png'),
                                                                 ellipse, confusion_point, luminances)
            diffs_arr.append(diffs)
            (dist_mean, dist_median, dist_std, dists) = dist_img(image_dir.joinpath(f'{name}-{tag}.png'),
                                                                 image_dir.joinpath(f'{name}-original.png'))
            dists_arr.append(dists)
            print(
                f'# {name}-{tag}: Difference {(diff_mean, diff_median, diff_std)}, Distance {(dist_mean, dist_median, dist_std)}')

        # aply statistical tests
        # kruskal_wallis_test from scipy

        calculate_statistical_test(name, "Distância", dists_arr)
        calculate_statistical_test(name, "Diferenciação", diffs_arr)
