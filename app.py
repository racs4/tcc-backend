import base64
import io
import math
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin
from ellipse import LsqEllipse
from webicd.convert import luv_chroma_to_luv_gama
from webicd.img import recolor_image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Rest of the code...

# =======================
# CONFIG

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# =======================
# CONSTANTS

CIRCLE = {
    'center': [-284.2029539, 244.51808],
    'r': math.sqrt(405439.059617)
}

# =======================
# ROUTES


@app.route("/image", methods=['POST'])
@cross_origin()
def image():
    """
    This route receives the image and returns the recolored image

    Returns:
        str
    """
    parameters = request.json
    image = recolor_image(parameters['image'], parameters['method'], parameters['ellipse'],
                          parameters['confusionPoint'], parameters['luminances'])
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    response = base64.b64encode(buffered.getvalue())
    return response


@app.route("/ellipse", methods=['Post'])
@cross_origin()
def get_ellipse():
    """
    This route receives the points and returns the ellipse parameters

    Returns:
        dict: {
            'parameters': {
                'width': float,
                'height': float,
                'center': [float, float],
                'phi': float
            },
            'coefficients': [float, float, float, float, float, float]
        }
    """
    points = request.json
    X = list(zip(points['x'], points['y']))
    X = list(map(lambda x: luv_chroma_to_luv_gama([50, x[0], x[1]])[1:], X))
    print(X)
    X = np.array(X)
    reg = LsqEllipse().fit(X)
    center, width, height, phi = reg.as_parameters()
    coefficients = reg.coefficients
    ellipse = {
        "parameters": {
            "width": width,
            "height": height,
            "center": center,
            "phi": phi
        },
        "coefficients": coefficients
    }
    return ellipse


@app.route("/confusion-point", methods=['Post'])
@cross_origin()
def get_confusion_point():
    """
    This route receives the ellipse parameters and returns the confusion point

    Returns:
        dict: {
            'confusionPoint': [
                [x1, y1],
                [x2, y2]
            ]
        }
    """
    parameters = request.json
    ellipse = parameters['ellipse']
    # y = m(x - x0) + y0
    # y = mx - mx0 + y0
    angular_line_coeficient = math.tan(ellipse['phi'] + math.pi/2)
    linear_line_coeficient = -(angular_line_coeficient *
                               ellipse['center'][0]) + ellipse['center'][1]

    # (x - t1)^2 + (y - t2)^2 = r^2
    # (x - t1)^2 + (mx - mx0 + y0 - t2)^2 = r^2
    # (x - t1)^2 + (mx - t2')^2 = r^2
    # (x^2 - 2xt1 + t1^2) + (m^2x^2 - 2mxt2' + t2'^2) = r^2
    # (1 + m^2)x^2 -(t1 + mt2')2x + t1^2 + t2'^2 - r^2 = 0
    new_t2 = -(linear_line_coeficient - CIRCLE['center'][1])
    a = 1 + math.pow(angular_line_coeficient, 2)
    b = -(2 * CIRCLE['center'][0]) - 2 * (angular_line_coeficient * new_t2)
    c = math.pow(CIRCLE['center'][0], 2) + \
        math.pow(new_t2, 2) - math.pow(CIRCLE['r'], 2)

    [x1, x2] = baskhara(a, b, c)
    return {'confusionPoint': [
        [x1, x1 * angular_line_coeficient + linear_line_coeficient],
        [x2, x2 * angular_line_coeficient + linear_line_coeficient],
    ]}


@app.route("/differentiation", methods=['Post'])
@cross_origin()
def differentiation():
    """
    This route receives the parameters and returns the differentiation

    Returns:
        dict: {
            'diff': float
        }
    """
    parameters = request.json
    ellipse = parameters['ellipse']
    colors = parameters['colors']
    confusion_point = parameters['confusionPoint']
    luminances = parameters['luminances']
    luminance_distance = (luminances['top'] - luminances['bottom']) / 2
    color1_luminance = colors['1'][0]
    color2_luminance = colors['2'][0]
    color_luminance_distance = abs(color1_luminance - color2_luminance)
    color1_point = colors['1'][1:]
    color2_point = colors['2'][1:]

    x = parameters['x']
    y = parameters['y']

    if (color_luminance_distance > luminance_distance):
        print("Luminance distance difference")
        return {"diff": color_luminance_distance / luminance_distance}
    else:
        print("Luminance distance difference")
        transformed_ellipse_angle = math.atan(
            (confusion_point[1] - color1_point[1]) /
            (confusion_point[0] - color1_point[0])
        ) + math.pi / 2
        transformed_ellipse_width = math.sqrt(math.pow(ellipse['width'], 2) - ((math.pow(
            ellipse['width'], 2) * math.pow(color_luminance_distance, 2))/math.pow(luminance_distance, 2)))
        transformed_ellipse_height = math.sqrt(math.pow(ellipse['height'], 2) - ((math.pow(
            ellipse['height'], 2) * math.pow(color_luminance_distance, 2))/math.pow(luminance_distance, 2)))
        transformed_ellipse = {
            'width': transformed_ellipse_width,
            'height': transformed_ellipse_height,
            'phi': transformed_ellipse_angle,
            'center': color1_point
        }
        # print(transformed_ellipse)
        p = ellipse_contains(transformed_ellipse, color2_point)
        # print(p)

        fig, ax = plt.subplots(1)
        ax.set_aspect("equal")

        plot_line(ax, ellipse['center'], [ellipse['center'][0], 0])
        # plot_confusion(ax, confusion_point)
        plot_fitted(ax, ellipse, x, y, confusion_point)
        plot_transformed(ax, transformed_ellipse, color1_point,
                         color2_point, confusion_point)

        fig.savefig('temp2.png', dpi=fig.dpi)

        return {"diff": p}

# =======================
# UTILS


def ellipse_contains(ellipse, point):
    """
    This function returns the value of the ellipse contains the point

    Args:
        ellipse (dict): The ellipse parameters
        point (list): The point

    Returns:
        float: The value of the ellipse contains the point
    """
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


def plot_confusion(ax, confusion_point):
    """
    This function plots the confusion point

    Args:
        ax (matplotlib.axes.Axes): The axes to plot the confusion point
        confusion_point (list): The confusion point
    """
    PROTANOPE_CONFUSION = [0.678, 0.501]
    DEUTERANOPE_CONFUSION = [-1.217, 0.782]
    TRITANOPE_CONFUSION = [0.257, 0.0]

    PROTANOPE_CONFUSION_AWAY = [-1.102, 0.380]
    DEUTERANOPE_CONFUSION_AWAY = [0.620, 0.375]
    TRITANOPE_CONFUSION_AWAY = [0.031, 1.786]

    g_circle = patches.Ellipse(
        CIRCLE['center'],
        2*CIRCLE['r'],
        2*CIRCLE['r'],
        fill=False, edgecolor='black', linestyle="dashdot", linewidth=2, alpha=0.5)
    ax.add_patch(g_circle)

    ax.plot(PROTANOPE_CONFUSION[0], PROTANOPE_CONFUSION[1], 'r*', alpha=0.7)
    ax.plot(DEUTERANOPE_CONFUSION[0],
            DEUTERANOPE_CONFUSION[1], 'g*', alpha=0.7)
    ax.plot(TRITANOPE_CONFUSION[0], TRITANOPE_CONFUSION[1], 'b*', alpha=0.7)

    ax.plot(PROTANOPE_CONFUSION_AWAY[0],
            PROTANOPE_CONFUSION_AWAY[1], 'r*', alpha=0.7)
    ax.plot(DEUTERANOPE_CONFUSION_AWAY[0],
            DEUTERANOPE_CONFUSION_AWAY[1], 'g*', alpha=0.7)
    ax.plot(TRITANOPE_CONFUSION_AWAY[0],
            TRITANOPE_CONFUSION_AWAY[1], 'b*', alpha=0.7)

    plot_line(ax, PROTANOPE_CONFUSION, PROTANOPE_CONFUSION_AWAY,
              color="red", linestyle="--", alpha=0.5)
    plot_line(ax, DEUTERANOPE_CONFUSION, DEUTERANOPE_CONFUSION_AWAY,
              color="green", linestyle="--", alpha=0.5)
    plot_line(ax, TRITANOPE_CONFUSION, TRITANOPE_CONFUSION_AWAY,
              color="blue", linestyle="--", alpha=0.5)

    ax.plot(confusion_point[0], confusion_point[1],
            'x', color="black", alpha=0.5)


def plot_fitted(ax, ellipse, x, y, confusion_point):
    """
    This function plots the fitted ellipse

    Args:
        ax (matplotlib.axes.Axes): The axes to plot the ellipse
        ellipse (dict): The ellipse parameters
        x (list): The x points
        y (list): The y points
        confusion_point (list): The confusion point
    """
    g_ellipse = patches.Ellipse(
        ellipse['center'],
        2*ellipse['width'],
        2*ellipse['height'],
        angle=(np.rad2deg(ellipse['phi'])),
        fill=False, edgecolor='red', linewidth=2, alpha=0.5)

    ax.add_patch(g_ellipse)
    ax.plot(x[0:2], y[0:2], 'r.', alpha=0.5)
    ax.plot(x[2:4], y[2:4], 'g.', alpha=0.5)
    ax.plot(x[4:6], y[4:6], 'b.', alpha=0.5)
    plot_line(ax, ellipse['center'], confusion_point,
              color="black", linestyle="dotted", alpha=0.3)


def plot_transformed(ax, transformed_ellipse, color1_point, color2_point, confusion_point):
    """
    This function plots the transformed ellipse

    Args:
        ax (matplotlib.axes.Axes): The axes to plot the ellipse
        transformed_ellipse (dict): The ellipse parameters
        color1_point (list): The first color point
        color2_point (list): The second color point
        confusion_point (list): The confusion
    """
    g_ellipse = patches.Ellipse(
        transformed_ellipse['center'],
        2*transformed_ellipse['width'],
        2*transformed_ellipse['height'],
        angle=np.rad2deg(transformed_ellipse['phi']),
        fill=False, edgecolor='green', linewidth=2, alpha=0.7)
    ax.add_patch(g_ellipse)
    ax.plot(color2_point[0], color2_point[1], 'bd')
    ax.plot(color1_point[0], color1_point[1], 'gd')
    plot_line(ax, transformed_ellipse['center'], confusion_point,
              color="black", linestyle="dotted", alpha=0.3)


def plot_line(ax, point1, point2, **kwargs):
    """
    This function plots a line between two points

    Args:
        ax (matplotlib.axes.Axes): The axes to plot the line
        point1 (list): The first point
        point2 (list): The second
        **kwargs: The line properties
    """
    ax.axline(point1, point2, **kwargs)


def baskhara(a, b, c):
    """
    This function solves the baskhara equation

    Args:
        a (float): The a coefficient
        b (float): The b coefficient
        c (float): The c coefficient

    Returns:
        list: The x values
    """
    delta = math.pow(b, 2) - 4 * a * c
    x1 = (-b + math.sqrt(delta)) / (2 * a)
    x2 = (-b - math.sqrt(delta)) / (2 * a)
    return [x1, x2]
