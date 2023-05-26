from distutils.util import execute

from webicd.icd import differentiation
from .abstract import AbstractFindConfusionColorStrategy
from igraph import Graph

class GraphGenerator(AbstractFindConfusionColorStrategy):

  def __init__(self, ellipse, confusion_point, luminances) -> None:
      self.ellipse = ellipse
      self.confusion_point = confusion_point
      self.luminances = luminances

  def execute(self, colors):
    g = Graph()

    for i in range(len(colors)):
      g.add_vertices([i])

    g.vs["color"] = colors

    for i in range(len(colors)):
      for j in range(i, len(colors)):
        if i != j:
          diff = differentiation(
            self.ellipse, 
            self.confusion_point, 
            self.luminances, 
            colors[i], colors[j]
          )
          if diff <= 1: # TODO ?
            g.add_edges([(i,j)])
            
    return g