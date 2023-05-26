from abc import ABC, abstractmethod

class AbstractRecolorStrategy(ABC):
  @abstractmethod
  def execute(self, image, colorDict, **kwargs):
    pass