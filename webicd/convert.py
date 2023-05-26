import math

# D65 white in XYZ
WHITEREF = [95.047, 100.00, 108.883]
EPS = 0.008856
K = 903.3


def rgb_to_xyz(rgb):
    r = rgb[0] / 255
    g = rgb[1] / 255
    b = rgb[2] / 255

    # Assume sRGB
    r = (((r + 0.055) / 1.055) ** 2.4) if r > 0.04045 else (r / 12.92)
    g = (((g + 0.055) / 1.055) ** 2.4) if g > 0.04045 else (g / 12.92)
    b = (((b + 0.055) / 1.055) ** 2.4) if b > 0.04045 else (b / 12.92)

    x = (r * 0.4124564) + (g * 0.3575761) + (b * 0.1804375)
    y = (r * 0.2126729) + (g * 0.7151522) + (b * 0.072175)
    z = (r * 0.0193339) + (g * 0.119192) + (b * 0.9503041)

    return (x * 100, y * 100, z * 100)


def xyz_to_rgb(xyz):
    x = xyz[0] / 100
    y = xyz[1] / 100
    z = xyz[2] / 100

    r = (x * 3.2404542) + (y * -1.5371385) + (z * -0.4985314)
    g = (x * -0.969266) + (y * 1.8760108) + (z * 0.041556)
    b = (x * 0.0556434) + (y * -0.2040259) + (z * 1.0572252)

    # Assume sRGB
    r = ((1.055 * (r ** (1.0 / 2.4))) - 0.055) if r > 0.0031308 else r * 12.92
    g = ((1.055 * (g ** (1.0 / 2.4))) - 0.055) if g > 0.0031308 else g * 12.92
    b = ((1.055 * (b ** (1.0 / 2.4))) - 0.055) if b > 0.0031308 else b * 12.92

    r = min(max(0, r), 1)
    g = min(max(0, g), 1)
    b = min(max(0, b), 1)

    return (r * 255, g * 255, b * 255)


def fmt(n):
    if math.isnan(n):
        return 0
    else:
        return n


def get_up(xyz):
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]
    if ((X + 15 * Y + 3 * Z) != 0):
        return 4 * X / (X + 15 * Y + 3 * Z)
    else:
        return 0


def get_vp(xyz):
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]
    if ((X + 15 * Y + 3 * Z) != 0):
        return 9 * Y / (X + 15 * Y + 3 * Z)
    else:
        return 0


def xyz_to_luv_chroma(xyz):
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]

    yr = Y / WHITEREF[1]
    up = get_up([X, Y, Z])
    vp = get_vp([X, Y, Z])

    L = fmt(116 * (yr ** (1 / 3)) - 16 if yr > EPS else K * yr)

    return (L, up, vp)


def luv_chroma_to_luv_gama(luv_chroma):
    L = luv_chroma[0]
    up = luv_chroma[1]
    vp = luv_chroma[2]

    upr = get_up(WHITEREF)
    vpr = get_vp(WHITEREF)

    u = 13 * L * (up - upr)
    v = 13 * L * (vp - vpr)

    return (L, u, v)


def luv_chroma_to_xyz(luv):
    upr = get_up(WHITEREF)
    vpr = get_vp(WHITEREF)

    L = luv[0]
    up = luv[1]
    vp = luv[2]

    u = 13 * L * (up - upr)
    v = 13 * L * (vp - vpr)

    Y = fmt(((L + 16) / 116) ** 3 if L > K * EPS else L / K)

    a = (1 / 3) * (((52 * L) / (u + 13 * L * upr)) - 1)
    b = -5 * Y
    c = -1 / 3
    d = Y * (((39 * L) / (v + 13 * L * vpr)) - 5)

    X = fmt((d - b) / (a - c))
    Z = fmt(X * a + b)
    return tuple(list(map(lambda x: x * 100, [X, Y, Z])))


def luv_gama_to_xyz(luv_gama):
    L = luv_gama[0]
    u = luv_gama[1]
    v = luv_gama[2]

    upr = get_up(WHITEREF)
    vpr = get_vp(WHITEREF)

    Y = fmt(((L + 16) / 116) ** 3 if L > K * EPS else L / K)

    a = (1 / 3) * (((52 * L) / (u + 13 * L * upr)) - 1)
    b = -5 * Y
    c = -1 / 3
    d = Y * (((39 * L) / (v + 13 * L * vpr)) - 5)

    X = fmt((d - b) / (a - c))
    Z = fmt(X * a + b)
    return tuple(list(map(lambda x: x * 100, [X, Y, Z])))


def rgb_to_luv_chroma(rgb):
    return xyz_to_luv_chroma(rgb_to_xyz(rgb))


def rgb_to_luv_gama(rgb):
    return luv_chroma_to_luv_gama(rgb_to_luv_chroma(rgb))


def luv_chroma_to_rgb(luv_chroma):
    return xyz_to_rgb(luv_chroma_to_xyz(luv_chroma))


def luv_gama_to_rgb(luv_gama):
    return xyz_to_rgb(luv_gama_to_xyz(luv_gama))


def convert_ar(ar, method):
    return list(map(method, ar))
