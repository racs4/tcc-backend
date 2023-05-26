from abc import ABC, abstractmethod

class AbstractRecolorConfusionColorStrategy(ABC):
  @abstractmethod
  def execute(self, colorGraph, **kwargs):
    pass