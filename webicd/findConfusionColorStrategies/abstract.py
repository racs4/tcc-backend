from abc import ABC, abstractmethod

class AbstractFindConfusionColorStrategy(ABC):
  @abstractmethod
  def execute(self, colors, **kwargs):
    pass