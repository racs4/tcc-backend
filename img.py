from PIL import Image, ImageOps, ImageFilter
import scipy.cluster
import sklearn.cluster
import numpy as np
import matplotlib.pyplot as plt


def dominant_colors(image):  # PIL image input
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # image = image.resize((150, 150))      # optional, to reduce time
    ar = np.asarray(image)
    ax.scatter(ar[:, 0], ar[:, 1], ar[:, 2], alpha=0.3, marker=".")
    # shape = ar.shape
    # ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    kmeans = sklearn.cluster.KMeans(
        # n_clusters=8,
        # init="random",
        # max_iter=2000,
        # random_state=1000,
        # max_no_improvement=None,
        # reassignment_ratio=0.5
    ).fit(ar)
    codes = kmeans.cluster_centers_
    ax.scatter(codes[:, 0], codes[:, 1], codes[:, 2],
               color="red", marker="o", alpha=1, s=100)
    plt.show()
    vecs, _dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, _bins = np.histogram(vecs, len(codes))    # count occurrences

    colors = []
    for index in np.argsort(counts)[::-1]:
        colors.append(tuple([int(code) for code in codes[index]]))

    # join labels
    labels = kmeans.labels_
    labels = list(
        map(
            lambda index: tuple([int(code) for code in codes[index]]),
            labels
        )
    )
    labeled = list(zip(ar, labels))

    return colors                    # returns colors in order of dominance

# def get_dominant_color(pil_img, palette_size=8):
#     # Resize image to speed up processing
#     img = pil_img.copy()
#     # img.thumbnail((100, 100))
#     # img = img.filter(ImageFilter.MedianFilter(3))

#     # Reduce colors (uses k-means internally)
#     paletted = img.convert('P', palette=Image.Palette.ADAPTIVE, colors=palette_size)

#     # Find the color that occurs most often
#     palette = paletted.getpalette()
#     color_counts = sorted(paletted.getcolors(), reverse=True)
#     palette_indexes = map(lambda x: x[1], color_counts)
#     dominant_colors = list(map(lambda x: palette[x*3:x*3+3], palette_indexes))

    # return dominant_colors


def get_colors_count(img, max=None):
    colors = img.getcolors(img.size[0]*img.size[1])
    colors = sorted(colors, reverse=True)
    if max is not None:
        colors = colors[0:max]
    return colors


def get_colors(img, max=None):
    return list(map(lambda x: x[1], get_colors_count(img, max)))


img = Image.open('flower.jpeg').convert('RGB')
colors = get_colors(img, max=10000)

# print(dominant_colors(colors))
# print(dominant_colors(im2))
# print(get_dominant_color(im2))
