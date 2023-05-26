from PIL.Image import Image
from webicd.convert import luv_gama_to_rgb, rgb_to_luv_gama
from .abstract import AbstractRecolorStrategy
import numpy as np


class RecolorCluster(AbstractRecolorStrategy):
    def __init__(self, model) -> None:
        self.model = model

    @staticmethod
    def translate(point, basePoint, newBasePoint):
        L = basePoint[0] - newBasePoint[0]
        u = basePoint[1] - newBasePoint[1]
        v = basePoint[2] - newBasePoint[2]
        return (point[0] + L, point[1] + u, point[2] + v)

    @staticmethod
    def recolor(model, color_dict):
        colors = model.cluster_centers_
        labels = model.predict()

        def recolor_pixel(pixel_color):
            # print(pixel_color)
            pixel_color_luv = rgb_to_luv_gama(pixel_color)
            cluster_label = model.predict(pixel_color_luv)
            label_color = colors[cluster_label]
            label_color_recolor = color_dict[label_color]
            new_pixel_color_luv = RecolorCluster.translate(
                pixel_color_luv, label_color, label_color_recolor)
            new_pixel_color = luv_gama_to_rgb(new_pixel_color_luv)
            return new_pixel_color
        return recolor_pixel

    @staticmethod
    def recolor2(colors, labels, color_dict, pivot_colors):
        '''
        Recolor a pixel based on its cluster label.
        '''
        def aux(idx):
            actual_color = colors[idx]
            pivot = pivot_colors[labels[idx]]
            pivot_recolor = color_dict[tuple(pivot)]
            recolor = RecolorCluster.translate(
                actual_color, pivot, pivot_recolor)
            return recolor
        return aux

    def execute(self, image: Image, color_dict):
        # get image array
        array = np.asarray(image)
        # print(array)
        # print("=================")
        # reshape to transform array in list of rgb colors
        # print(array.shape)
        (m, n, k) = array.shape
        array = np.reshape(array, (m*n, k))

        # transform to luv
        array = np.apply_along_axis(rgb_to_luv_gama, 1, array)
        # print(array)
        # print("=================")
        # predict color for all arrays
        labels = self.model.predict(array)
        pivot_colors = self.model.cluster_centers_

        array = np.array(list(map(RecolorCluster.recolor2(
            array, labels, color_dict, pivot_colors), range(len(array)))))

        # print(array)
        # print("=================")
        array = np.apply_along_axis(luv_gama_to_rgb, 1, array)
        array = np.reshape(array, (m, n, k))

        # for i in range(len(array)):
        #   row = array[i]
        #   for j in range(len(row)):
        #     pixel_color = array[i,j]
        #     # print(pixel_color)
        #     pixel_color_luv = rgb_to_luv(pixel_color)
        #     cluster_label = self.model.predict([pixel_color_luv])
        #     label_color = tuple(colors[cluster_label][0])
        #     label_color_recolor = color_dict[label_color]
        #     new_pixel_color_luv = RecolorCluster.translate(pixel_color_luv, label_color, label_color_recolor)
        #     new_pixel_color = luv_chroma_to_rgb(new_pixel_color_luv)
        #     new_pixel_color = list(map(lambda x: int(x), new_pixel_color))
        #     # print(pixel_color)
        #     # print(new_pixel_color)
        #     array[i,j] = new_pixel_color
        #     # print(array[i,j])
        #     # print(pixel_color)

        # print(array)
        # print(array.shape)
        array = array.astype('uint8')
        # print("=======================")
        # print(array)
        # print(array.shape)
        return array
