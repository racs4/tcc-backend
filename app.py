import math
from flask import Flask, request
from flask_cors import CORS, cross_origin
from ellipse import LsqEllipse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from colour.plotting import plot_chromaticity_diagram_CIE1976UCS

# =======================
# CONFIG

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# =======================
# CONSTANTS

CIRCLE = {
  'center': [-0.2393954889781, 0.844517965788],
  'r': math.sqrt(0.9596190760166)
}


# =======================
# ROUTES

@app.route("/ellipse", methods=['Post'])
@cross_origin()
def getEllipse():
    points = request.json
    X = np.array(list(zip(points['x'], points['y'])))
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
def getConfusionPoint():
    parameters = request.json
    ellipse = parameters['ellipse']
    # y = m(x - x0) + y0
    # y = mx - mx0 + y0
    print(ellipse)
    print(ellipse['phi'])
    print(np.rad2deg(ellipse['phi']))
    angularLineCoef = math.tan(ellipse['phi'] + math.pi/2)
    print(angularLineCoef)
    linearLineCoef = -(angularLineCoef * ellipse['center'][0]) + ellipse['center'][1]

    # (x - t1)^2 + (y - t2)^2 = r^2
    # (x - t1)^2 + (mx - mx0 + y0 - t2)^2 = r^2
    # (x - t1)^2 + (mx - t2')^2 = r^2
    # (x^2 - 2xt1 + t1^2) + (m^2x^2 - 2mxt2' + t2'^2) = r^2
    # (1 + m^2)x^2 -(t1 + mt2')2x + t1^2 + t2'^2 - r^2 = 0
    new_t2 = -(linearLineCoef - CIRCLE['center'][1])
    a = 1 + math.pow(angularLineCoef, 2)
    b = -(2 * CIRCLE['center'][0]) - 2 * (angularLineCoef * new_t2)
    c = math.pow(CIRCLE['center'][0], 2) + math.pow(new_t2, 2) - math.pow(CIRCLE['r'], 2)

    [x1, x2] = baskhara(a, b, c)
    return {'confusionPoint': [
      [x1, x1 * angularLineCoef + linearLineCoef],
      [x2, x2 * angularLineCoef + linearLineCoef],
    ]}

@app.route("/differentiation", methods=['Post'])
@cross_origin()
def differentiation():
    parameters = request.json
    ellipse = parameters['ellipse']
    colors = parameters['colors']
    confusion_point = parameters['confusionPoint']
    luminances = parameters['luminances']
    luminance_distance = (luminances['top'] - luminances['bottom']) / 2
    color1Luminance = colors['1'][0]
    color2Luminance = colors['2'][0]
    color_luminance_distance = abs(color1Luminance - color2Luminance)
    color1Point = colors['1'][1:]
    color2Point = colors['2'][1:]

    x = parameters['x']
    y = parameters['y']


    if (color_luminance_distance > luminance_distance):
        return {"diff": color_luminance_distance / luminance_distance}
    else:
        transformed_ellipse_angle =  math.atan(
            (confusion_point[1] - color1Point[1]) / 
            (confusion_point[0] - color1Point[0])
        ) + math.pi / 2
        transformed_ellipse_width = math.sqrt(math.pow(ellipse['width'], 2) - ((math.pow(ellipse['width'], 2) * math.pow(color_luminance_distance, 2))/math.pow(luminance_distance, 2)))
        transformed_ellipse_height = math.sqrt(math.pow(ellipse['height'], 2) - ((math.pow(ellipse['height'], 2) * math.pow(color_luminance_distance, 2))/math.pow(luminance_distance, 2)))
        transformed_ellipse = {
            'width': transformed_ellipse_width,
            'height': transformed_ellipse_height,
            'phi': transformed_ellipse_angle,
            'center': color1Point
        }
        print(transformed_ellipse)
        p = ellipse_contains(transformed_ellipse, color2Point)
        # print(p)

        fig,ax = plt.subplots(1)
        ax.set_aspect("equal")
        
        plot_line(ax, ellipse['center'], [ellipse['center'][0], 0])
        # plot_confusion(ax, confusion_point)
        plot_fitted(ax, ellipse, x, y, confusion_point)
        plot_transformed(ax, transformed_ellipse, color1Point, color2Point, confusion_point)
        
        fig.savefig('temp2.png', dpi=fig.dpi)
        

        return {"diff": p}

# =======================
# METHODS

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

def plot_confusion(ax, confusion_point):
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
    ax.plot(DEUTERANOPE_CONFUSION[0], DEUTERANOPE_CONFUSION[1], 'g*', alpha=0.7)
    ax.plot(TRITANOPE_CONFUSION[0], TRITANOPE_CONFUSION[1], 'b*', alpha=0.7)

    ax.plot(PROTANOPE_CONFUSION_AWAY[0], PROTANOPE_CONFUSION_AWAY[1], 'r*', alpha=0.7)
    ax.plot(DEUTERANOPE_CONFUSION_AWAY[0], DEUTERANOPE_CONFUSION_AWAY[1], 'g*', alpha=0.7)
    ax.plot(TRITANOPE_CONFUSION_AWAY[0], TRITANOPE_CONFUSION_AWAY[1], 'b*', alpha=0.7)

    plot_line(ax, PROTANOPE_CONFUSION, PROTANOPE_CONFUSION_AWAY, color="red", linestyle="--", alpha=0.5)
    plot_line(ax, DEUTERANOPE_CONFUSION, DEUTERANOPE_CONFUSION_AWAY, color="green", linestyle="--", alpha=0.5)
    plot_line(ax, TRITANOPE_CONFUSION, TRITANOPE_CONFUSION_AWAY, color="blue", linestyle="--", alpha=0.5)

    ax.plot(confusion_point[0], confusion_point[1], 'x', color="black", alpha=0.5)

def plot_fitted(ax, ellipse, x, y, confusion_point):
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
    plot_line(ax, ellipse['center'], confusion_point, color="black", linestyle="dotted", alpha=0.3)

def plot_transformed(ax, transformed_ellipse, color1Point, color2Point, confusion_point):
    g_ellipse = patches.Ellipse(
        transformed_ellipse['center'], 
        2*transformed_ellipse['width'], 
        2*transformed_ellipse['height'], 
        angle=np.rad2deg(transformed_ellipse['phi']), 
        fill=False, edgecolor='green', linewidth=2, alpha=0.7)
    ax.add_patch(g_ellipse)
    ax.plot(color2Point[0],color2Point[1],'bd')
    ax.plot(color1Point[0],color1Point[1],'gd')
    plot_line(ax, transformed_ellipse['center'], confusion_point, color="black", linestyle="dotted", alpha=0.3)

def plot_line(ax, point1, point2, **kwargs):
    # slope = math.tan(math.pi/2 + 0.00000000001) if point2[0] == point1[0] else (point2[1] - point1[1]) / (point2[0] - point1[0])
    ax.axline(point1, point2, **kwargs)
    # line = lines.Line2D(x_values, y_values, **kwargs)
    # ax.add_line(line)


def baskhara(a, b, c):
  delta = math.pow(b, 2) - 4 * a * c
  x1 = (-b + math.sqrt(delta)) / (2 * a)
  x2 = (-b - math.sqrt(delta)) / (2 * a)
  return [x1, x2];