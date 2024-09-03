import numpy as np
import math

from webicd.utils import sigmoid


def differentiation(
    ellipse,
    confusion_point,
    luminances,
    color1,
    color2
):
    [L, u, v] = color1
    # transformed_ellipse_angle = math.atan(
    #         (confusion_point[1] - v) /
    #         (confusion_point[0] - u)
    #     ) + math.pi / 2
    # ellipse['phi']  = transformed_ellipse_angle
    return sigmoid(color1, color2, ellipse, luminances)
    # luminance_distance = (luminances['top'] - luminances['bottom']) / 2
    # color1Luminance = color1[0]
    # color2Luminance = color2[0]
    # color_luminance_distance = abs(color1Luminance - color2Luminance)
    # color1Point = color1[1:]
    # color2Point = color2[1:]

    # if (color_luminance_distance > luminance_distance):
    #     return color_luminance_distance / luminance_distance
    # else:
    #     transformed_ellipse_angle = math.atan(
    #         (confusion_point[1] - color1Point[1]) /
    #         (confusion_point[0] - color1Point[0])
    #     ) + math.pi / 2
    #     transformed_ellipse_width = math.sqrt(math.pow(ellipse['width'], 2) - ((math.pow(
    #         ellipse['width'], 2) * math.pow(color_luminance_distance, 2))/math.pow(luminance_distance, 2)))
    #     transformed_ellipse_height = math.sqrt(math.pow(ellipse['height'], 2) - ((math.pow(
    #         ellipse['height'], 2) * math.pow(color_luminance_distance, 2))/math.pow(luminance_distance, 2)))
    #     transformed_ellipse = {
    #         'width': transformed_ellipse_width,
    #         'height': transformed_ellipse_height,
    #         'phi': transformed_ellipse_angle,
    #         'center': color1Point
    #     }
    #     p = ellipse_contains(transformed_ellipse, color2Point)
    #     return p


def differentiantion_ar(
    ellipse,
    confusion_point,
    luminances,
    color,
    ar
):
    """
    Returns True if the differentiation of the ellipse with the 
    confusion point and the luminances is greater than 1 for all the colors in the array

    Returns:
       Boolean  
    """
    return all(map(lambda x: differentiation(ellipse, confusion_point, luminances, color, x) > 1, ar))


def ellipse_contains(ellipse, point):
    cos_angle = np.cos(np.radians(90.-np.rad2deg(ellipse['phi'])))
    sin_angle = np.sin(np.radians(90.-np.rad2deg(ellipse['phi'])))

    x = point[0]
    y = point[1]

    xc = x - ellipse['center'][0]
    yc = y - ellipse['center'][1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct**2/(ellipse['width'])**2) + (yct**2/(ellipse['height'])**2)

    return rad_cc


def differentiantion_mean(
    ellipse,
    confusion_point,
    luminances,
    color,
    ar
):
    return np.mean(
        list(
            map(
                lambda x: differentiation(
                    ellipse, confusion_point, luminances, color, x),
                ar
            )
        )
    )
