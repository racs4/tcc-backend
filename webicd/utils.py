import math
from random import randint, random, seed, uniform

import numpy as np
from PIL.Image import Image

from webicd.convert import (WHITEREF, luv_chroma_to_xyz, luv_gama_to_xyz,
                            rgb_to_luv_chroma)


def get_colors_count(image: Image, max=None):
    """
    Receives an image and returns a list of tuples with the colors and their count.

    Args:
        image (Image): The input image.
        max (_type_, optional): The maximum number of colors to return. Defaults to None.
    Returns:
        list[tuple[int, int]]: A list of tuples in the format (count, color).
    """
    colors = image.getcolors(image.size[0]*image.size[1])
    colors = sorted(colors, reverse=True)
    if max is not None:
        colors = colors[0:max]
    return colors


def get_colors(image: Image, max=None):
    """
    Receives an image and returns a list of the colors.

    Args:
        image (Image): The input image.
        max (_type_, optional): The maximum number of colors to return. Defaults to None.
    Returns:
        list[int]: A list of the colors.
    """
    return list(map(lambda x: x[1], get_colors_count(image, max)))


def random_rgb(random_state=None):
    """
    Generates a random RGB color.

    Args:
        random_state (int, optional): The random state. Defaults to None.
    Returns:
        tuple[int, int, int]: A tuple with the RGB color.
    """
    seed(random_state)
    R = randint(0, 255)
    G = randint(0, 255)
    B = randint(0, 255)
    return (R, G, B)


def random_luv(random_state=None):
    """
    Generates a random LUV color.

    Args:
        random_state (int, optional): The random state. Defaults to None.
    Returns:
        list[float]: A list with the LUV color.
    """
    return rgb_to_luv_chroma(random_rgb(random_state))


def is_visible_luv_chroma(luv):
    """
    Checks if a LUV color is visible.

    Args:
        luv (list[float]): The LUV color.
    Returns:
        bool: True if the color is visible, False otherwise.
    """
    xyz = luv_chroma_to_xyz(luv)
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]
    return 0 <= X and X <= WHITEREF[0] and 0 <= Y and Y <= WHITEREF[1] and 0 <= Z and Z <= WHITEREF[2]


def is_visible_luv_gama(luv):
    """
    Checks if a LUV color is visible.

    Args:
        luv (list[float]): The LUV color.
    Returns:
        bool: True if the color is visible, False otherwise.
    """
    xyz = luv_gama_to_xyz(luv)
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]
    return 0 <= X and X <= WHITEREF[0] and 0 <= Y and Y <= WHITEREF[1] and 0 <= Z and Z <= WHITEREF[2]


def random_luv_chroma(luminance):
    """
    Generates a random LUV color with a given luminance.

    Args:
        luminance (float): The luminance.
    Returns:
        list[float]: A list with the LUV color.
    """
    u = random() * 0.7
    v = random() * 0.6
    while (not is_visible_luv_chroma([luminance, u, v])):
        u = random() * 0.7
        v = random() * 0.6
    return [luminance, u, v]


def color_distance(color_a, color_b):
    """
    Calculates the distance between two colors.

    Args:
        color_a (list[float]): The first color.
        color_b (list[float]): The second color.
    Returns:
        float: The distance between the two colors.
    """
    [a1, b1, c1] = color_a
    [a2, b2, c2] = color_b
    return math.sqrt(math.pow(a1 - a2, 2) + math.pow(b1 - b2, 2) + math.pow(c1 - c2, 2))


def translate_to_origin(point, center, phi):
    """
    Translates a point to the origin.

    Args:
        point (list[float]): The point.
        center (list[float]): The center.
        phi (float): The angle.
    Returns:
        list[float]: The translated point.
    """
    cos_angle = np.cos(np.radians(90.-np.rad2deg(phi)))
    sin_angle = np.sin(np.radians(90.-np.rad2deg(phi)))

    [L, u, v] = point
    [cl, cu, cv] = center

    uc = u - cu
    vc = v - cv
    lc = L - cl

    uct = uc * cos_angle - vc * sin_angle
    vct = uc * sin_angle + vc * cos_angle

    return [lc, uct, vct]


def ellipsoid_intersection(point, ellipse, luminances):
    """
    Calculates the intersection of an ellipsoid with a point.

    Args:
        point (list[float]): The point.
        ellipse (dict): The ellipse.
        luminances (dict): The luminances.
    Returns:
        list[float]: The intersection.
    """
    [L, u, v] = point
    [a, b] = [ellipse['width'], ellipse['height']]
    c = (luminances['top'] - luminances['bottom']) / 2
    t = 1 / math.sqrt(u ** 2 / a ** 2 + v ** 2 / b ** 2 + L ** 2 / c ** 2)
    return [t * L, t * u, t * v]


def sigmoid(c1, c2, ellipse, luminances):
    """
    Calculates the sigmoid function.

    Args:
        c1 (list[float]): The first color.
        c2 (list[float]): The second color.
        ellipse (dict): The ellipse.
        luminances (dict): The luminances.
    Returns:
        float: The sigmoid value.
    """
    c2 = translate_to_origin(c2, c1, ellipse['phi'])
    if c2 == [0, 0, 0]:
        return 0
    else:
        c1 = translate_to_origin(c1, c1, ellipse['phi'])
        i = ellipsoid_intersection(c2, ellipse, luminances)
        p5 = color_distance(c1, i)
        x = color_distance(c1, c2)
        return x / p5
