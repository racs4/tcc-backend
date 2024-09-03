from .findConfusionColorStrategies.abstract import AbstractFindConfusionColorStrategy
from .recolorConfusionColorStrategies.abstract import AbstractRecolorConfusionColorStrategy
from .recolorStrategies.abstract import AbstractRecolorStrategy
from .getColorStrategies.abstract import AbstractGetColorStrategy


class Recolor:
    def __init__(
        self,
        get_color_strategy: AbstractGetColorStrategy,
        find_confusion_color_strategy: AbstractFindConfusionColorStrategy,
        recolor_confusion_color_strategy: AbstractRecolorConfusionColorStrategy,
        recolor_strategy: AbstractRecolorStrategy
    ):
        """
        Initialize the recoloring algorithm

        Args:
            get_color_strategy (AbstractGetColorStrategy): The strategy to get the colors of the image
            find_confusion_color_strategy (AbstractFindConfusionColorStrategy): The strategy to find the confusion colors
            recolor_confusion_color_strategy (AbstractRecolorConfusionColorStrategy): The strategy to recolor the confusion colors
            recolor_strategy (AbstractRecolorStrategy): The strategy to recolor the image
        Returns:
            None
        """
        self.get_color_strategy = get_color_strategy
        self.find_confusion_color_strategy = find_confusion_color_strategy
        self.recolor_confusion_color_strategy = recolor_confusion_color_strategy
        self.recolor_strategy = recolor_strategy

    def execute(self, image):
        """
        Execute the recoloring process on an image

        Args:
            image (PIL.Image): The input image
        Returns:
            tuple[PIL.Image, dict]: A tuple containing the recolored image and a dictionary mapping the original colors to the new colors
        """
        colors = self.get_color_strategy.execute(image)
        color_graph = self.find_confusion_color_strategy.execute(colors)
        color_dict = self.recolor_confusion_color_strategy.execute(color_graph)
        new_image = self.recolor_strategy.execute(image, color_dict)
        return (new_image, color_dict)
