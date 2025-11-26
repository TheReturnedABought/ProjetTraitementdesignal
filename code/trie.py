#trie
import numpy as np
import skimage
from matplotlib import pyplot as plt


def detect_layout_from_image(img_paths):
    """
    Detect AZERTY vs QWERTY using signal processing on images list.
    Returns a list of detections per image.
    """
    results = []

    for img_path in img_paths:
        img = plt.imread(img_path)
        if img.shape[-1] == 4:
            img = img[..., :3]
        gray = skimage.color.rgb2gray(img)

        # Seuillage
        thresh = skimage.filters.threshold_otsu(gray)
        binary = gray < thresh

        # Nettoyage
        binary = skimage.morphology.remove_small_objects(binary, min_size=300)

        # Label connected components
        labels = skimage.measure.label(binary)
        regions = skimage.measure.regionprops(labels)

        # Touches significatives
        key_regions = [r for r in regions if r.area > 1000]
        if len(key_regions) < 10:
            results.append("Unknown")
            continue

        # Centroides
        centroids = np.array([r.centroid for r in key_regions])

        # Trier par y (ligne) puis par x (colonne)
        centroids_sorted_y = centroids[centroids[:, 0].argsort()]
        first_row = centroids_sorted_y[:10]
        first_row_sorted_x = first_row[first_row[:, 1].argsort()]

        xs = first_row_sorted_x[:, 1]
        diffs = np.diff(xs)

        if diffs[0] > np.mean(diffs) * 1.3:
            results.append("QWERTY")
        elif diffs[0] < np.mean(diffs) * 0.7:
            results.append("AZERTY")
        else:
            results.append("Unknown")

    return results