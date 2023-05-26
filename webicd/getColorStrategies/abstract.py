from abc import ABC, abstractmethod

class AbstractGetColorStrategy(ABC):

  @abstractmethod
  def execute(self, image, **kwargs):
    pass