import math
from random import randint, random, uniform, seed
from PIL.Image import Image
import numpy as np
from webicd.convert import luv_gama_to_xyz
from webicd.convert import rgb_to_luv_chroma, luv_chroma_to_xyz, WHITEREF


def get_colors_count(image: Image, max=None):
    colors = image.getcolors(image.size[0]*image.size[1])
    colors = sorted(colors, reverse=True)
    if max is not None:
        colors = colors[0:max]
    return colors


def get_colors(image: Image, max=None):
    return list(map(lambda x: x[1], get_colors_count(image, max)))


def random_rgb(random_state=None):
    seed(random_state)
    R = randint(0, 255)
    G = randint(0, 255)
    B = randint(0, 255)
    return (R, G, B)


def random_luv(random_state=None):
    return rgb_to_luv_chroma(random_rgb(random_state))


def is_visible_luv_chroma(luv):
    xyz = luv_chroma_to_xyz(luv)
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]
    return 0 <= X and X <= WHITEREF[0] and 0 <= Y and Y <= WHITEREF[1] and 0 <= Z and Z <= WHITEREF[2]

def is_visible_luv_gama(luv):
    xyz = luv_gama_to_xyz(luv)
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]
    return 0 <= X and X <= WHITEREF[0] and 0 <= Y and Y <= WHITEREF[1] and 0 <= Z and Z <= WHITEREF[2]


def random_luv_chroma(luminance):
    u = random() * 0.7
    v = random() * 0.6
    while (not is_visible_luv_chroma([luminance, u, v])):
        u = random() * 0.7
        v = random() * 0.6
    return [luminance, u, v]


def color_distance(colorA, colorB):
    [a1, b1, c1] = colorA
    [a2, b2, c2] = colorB
    return math.sqrt(math.pow(a1 - a2, 2) + math.pow(b1 - b2, 2) + math.pow(c1 - c2, 2))


def translate_to_origin(point, center, phi):
    cos_angle = np.cos(np.radians(90.-np.rad2deg(phi)))
    sin_angle = np.sin(np.radians(90.-np.rad2deg(phi)))

    [L, u, v] = point
    [cL, cu, cv] = center

    uc = u - cu
    vc = v - cv
    Lc = L - cL

    uct = uc * cos_angle - vc * sin_angle
    vct = uc * sin_angle + vc * cos_angle

    return [Lc, uct, vct]


def ellipsoid_intersection(point, ellipse, luminances):
    [L, u, v] = point
    [a, b] = [ellipse['width'], ellipse['height']]
    c = (luminances['top'] - luminances['bottom']) / 2
    t = 1 / math.sqrt(u ** 2 / a ** 2 + v ** 2 / b ** 2 + L ** 2 / c ** 2)
    return [t * L, t * u, t * v]

def sigmoid(c1, c2, ellipse, luminances):
    c2 = translate_to_origin(c2, c1, ellipse['phi'])
    if c2 == [0,0,0]:
        return 0
    else:
        c1 = translate_to_origin(c1, c1, ellipse['phi'])
        i = ellipsoid_intersection(c2, ellipse, luminances)
        p5 = color_distance(c1, i)
        x = color_distance(c1, c2)
        return x / p5
    # return 1 / (1 + math.exp(-0.1 * (x - p5)))
    
