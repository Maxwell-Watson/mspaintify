import os

import skimage.color as skcolour
from math import sqrt
from math import floor
from numpy import array
from skimage.transform import resize_local_mean
from skimage.io import imread
import skimage.data as data
from skimage.segmentation import slic
from matplotlib import pyplot as plt
from data import get_palette


def get_colour(ndarray):
    return ndarray[0], ndarray[1], ndarray[2]


def rgb_euclidean_distance(c1, c2):
    return sqrt(sum([abs(dist) for dist in [(a ^ 2) - (b ^ 2) for a, b in zip(c1, c2)]]))


def generate_colour_map(source_palette, target_palette):
    colour_map = {}
    for colour in source_palette:
        distances = [(rgb_euclidean_distance(colour, c2), c2) for c2 in target_palette]
        colour_map[colour] = sorted(distances, key=lambda x: x[0])[0][1]

    return colour_map


def segment_image(user_image, n_segments):
    resized = resize_local_mean(user_image, (200, 200), preserve_range=True)
    image = []

    for row in resized:
        new_row = []
        for pixel in row:
            new_row.append((floor(pixel[0]), floor(pixel[1]), floor(pixel[2])))
        image.append(new_row)
    segment_labels = slic(image, n_segments=50, compactness=1, enforce_connectivity=False)
    return skcolour.label2rgb(segment_labels, image, kind="avg", bg_label=0)


def find_palette(image):
    segmented_palette = set()
    for row in image:
        for pixel in row:
            segmented_palette.add(get_colour(pixel))
    return segmented_palette


def transform_colours(image, colour_map):
    transformed = []
    for row in image:
        new_row = []
        for pixel in row:
            new_row.append(colour_map[get_colour(pixel)])
        transformed.append(new_row)
    return transformed


def main(user_image, palette_name):
    target_palette = get_palette(palette_name)
    segmented_image = segment_image(user_image, len(target_palette))
    segmented_palette = find_palette(segmented_image)
    colour_map = generate_colour_map(segmented_palette, target_palette)
    return array(transform_colours(segmented_image, colour_map))


# Script
image = data.astronaut()  # Insert your own here with imread({path-to-file})
xp_image = main(image, "xp")
vista_image = main(image, "vista")
seven_image = main(image, "seven")
hybrid_image = main(image, "all")

f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(xp_image)
axarr[0, 1].imshow(vista_image)
axarr[1, 0].imshow(seven_image)
axarr[1, 1].imshow(hybrid_image)
plt.show()
