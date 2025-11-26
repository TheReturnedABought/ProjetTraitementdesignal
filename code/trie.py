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
            results.append(describe_unknown_keyboard(key_regions))

    return results

def describe_unknown_keyboard(key_regions):
    centroids = [r.centroid for r in key_regions]
    sizes = [r.area for r in key_regions]
    distrib = "left/right split" if centroids and max(c[1] for c in centroids) - min(c[1] for c in centroids) > 300 else "normal/compact"

    # Heuristique QWERTY/AZERTY basée sur la première rangée (algorithme spatial)
    # 1. Extraire la rangée supérieure (y minimal)
    if centroids:
        ys = np.array([c[0] for c in centroids])
        min_y = np.min(ys)
        # On prend les touches dans la zone supérieure (y proche du minimum)
        row_indices = np.where(np.abs(ys - min_y) < 30)[0]
        row_keys = [centroids[i] for i in row_indices]
        if len(row_keys) >= 6:
            row_keys = sorted(row_keys, key=lambda c: c[1])
            xs = [c[1] for c in row_keys]
            diffs = np.diff(xs)
            mean_diff = np.mean(diffs[1:]) if len(diffs) > 1 else 0
            if mean_diff > 0:
                if diffs[0] > mean_diff * 1.3:
                    layout_guess = "Probably QWERTY (large left offset)"
                elif diffs[0] < mean_diff * 0.7:
                    layout_guess = "Probably AZERTY (small left offset)"
                else:
                    layout_guess = "Layout unknown (row spacing analysis)"
            else:
                layout_guess = "Not enough spacing for layout heuristics"
        else:
            layout_guess = "Not enough keys in top row"
    else:
        layout_guess = "No keys detected"

    return {
        "status": "Unknown keyboard layout",
        "detected_keys": len(key_regions),
        "distribution": distrib,
        "layout_guess": layout_guess,
        "example_centroids": centroids[:5],
        "sizes": sizes[:5]
    }
