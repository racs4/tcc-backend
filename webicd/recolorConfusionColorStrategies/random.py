from random import seed
from webicd.icd import differentiantion_ar
from webicd.utils import random_luv, random_luv_chroma
from .abstract import AbstractRecolorConfusionColorStrategy


class RandomRecolor(AbstractRecolorConfusionColorStrategy):
    def __init__(self, ellipse, confusion_point, luminances, random_state=None, max_steps=10000) -> None:
        self.ellipse = ellipse
        self.confusion_point = confusion_point
        self.luminances = luminances
        self.random_state = random_state
        self.max_steps = max_steps

    def execute(self, colorGraph):
        print('Executing RandomRecolor')
        seed(self.random_state)
        color_dict = {}
        colors = colorGraph.vs["color"]

        # while there is some node with degree greater than 0
        while len(colorGraph.vs(_degree_gt=0)) > 0:
            vertices = colorGraph.vs.select(_degree=colorGraph.maxdegree())
            v = vertices[0]
            v_color = v["color"]
            print(f'v_color: {v_color}')

            new_color = random_luv(random_state=None)
            index = v.index
            all_diff = differentiantion_ar(
                self.ellipse,
                self.confusion_point,
                self.luminances,
                new_color, colors[:index]+colors[index+1:]
            )
            step = 0
            while not all_diff and self.max_steps > step:
                print(f"{v_color} -> {new_color}, {step}")
                new_color = random_luv(random_state=None)
                all_diff = differentiantion_ar(
                    self.ellipse,
                    self.confusion_point,
                    self.luminances,
                    new_color, colors[:index]+colors[index+1:]
                )
                step += 1

            colorGraph.delete_edges(v.all_edges())
            if (all_diff):
                color_dict[v_color] = new_color
            else:
                color_dict[v_color] = v_color
            

        for vertice in colorGraph.vs:
            color = vertice["color"]
            if color_dict.get(color) is None:
                color_dict[vertice["color"]] = vertice["color"]

        return color_dict
