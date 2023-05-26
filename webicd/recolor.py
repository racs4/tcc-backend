from .findConfusionColorStrategies.abstract import AbstractFindConfusionColorStrategy
from .recolorConfusionColorStrategies.abstract import AbstractRecolorConfusionColorStrategy
from .recolorStrategies.abstract import AbstractRecolorStrategy
from .getColorStrategies.abstract import AbstractGetColorStrategy


class Recolor:
  def __init__(
    self, 
    getColorStrategy: AbstractGetColorStrategy, 
    findConfusionColorStrategy: AbstractFindConfusionColorStrategy,
    recolorConfusionColorStrategy: AbstractRecolorConfusionColorStrategy,
    recolorStrategy: AbstractRecolorStrategy
  ):
    self.getColorStrategy = getColorStrategy
    self.findConfusionColorStrategy = findConfusionColorStrategy
    self.recolorConfusionColorStrategy = recolorConfusionColorStrategy
    self.recolorStrategy = recolorStrategy

  def execute(self, image):
    colors = self.getColorStrategy.execute(image)
    colorGraph = self.findConfusionColorStrategy.execute(colors)
    colorDict = self.recolorConfusionColorStrategy.execute(colorGraph)
    newImage =  self.recolorStrategy.execute(image, colorDict)
    return (newImage, colorDict)