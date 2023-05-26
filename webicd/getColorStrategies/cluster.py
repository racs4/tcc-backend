from webicd.convert import rgb_to_luv_gama, convert_ar
from webicd.getColorStrategies.abstract import AbstractGetColorStrategy
from webicd.utils import get_colors
import numpy as np
import scipy.cluster
import sklearn.cluster


class Cluster(AbstractGetColorStrategy):
    def __init__(self, model, resize=False, maxColors=10000) -> None:
        self.model = model
        self.resize = resize
        self.maxColors = maxColors

    def execute(self, image):
        if self.resize:
            image = image.resize((150, 150))      # optional, to reduce time

        colors = get_colors(image, self.maxColors)
        colors = np.asarray(convert_ar(colors, rgb_to_luv_gama))
        colors = colors[~np.isnan(colors).any(axis=1), :]
        ar = np.asarray(colors)

        model = self.model.fit(ar)
        codes = model.cluster_centers_
        vecs, _dist = scipy.cluster.vq.vq(ar, codes)   # assign codes
        counts, _bins = np.histogram(vecs, len(codes))  # count occurrences

        colors = []
        for index in np.argsort(counts)[::-1]:
            colors.append(tuple([code for code in codes[index]]))

        # join labels
        labels = model.labels_
        labels = list(
            map(
                lambda index: tuple([code for code in codes[index]]),
                labels
            )
        )
        labeled = list(zip(ar, labels))

        return colors
