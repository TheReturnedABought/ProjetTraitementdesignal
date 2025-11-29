#trie
import skimage
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pytesseract
from util import sobel

def detect_layout_from_image(img_paths, debug = False):
    """
    Detect AZERTY vs QWERTY using signal processing on images list.
    Returns a list of detections per image.
    """
    results = []

    for img_path in img_paths:
        fig, axes = plt.subplots(1, 8, figsize=(25, 6))
        img = plt.imread(img_path)
        if img.shape[-1] == 4:
            img = img[..., :3]

        if debug == True:
            axes[0].imshow(img)
            axes[0].set_title("1. Original")
            axes[0].axis("off")

        gray = skimage.color.rgb2gray(img)
        edges = sobel(gray)

        if debug == True:
            axes[1].imshow(gray, cmap="gray")
            axes[1].set_title("2. Grayscale")
            axes[1].axis("off")

        if debug == True:
            axes[2].imshow(edges, cmap="gray")
            axes[2].set_title("3. Sobel")
            axes[2].axis("off")

        # Seuillage
        thresh = skimage.filters.threshold_otsu(edges)
        binary = edges < thresh

        if debug == True:
            axes[3].imshow(binary, cmap="gray")
            axes[3].set_title("4. Binary (Otsu)")
            axes[3].axis("off")

        # Label connected components
        labels = skimage.measure.label(binary)
        regions = skimage.measure.regionprops(labels)

        if debug == True:
            axes[4].imshow(labels, cmap="nipy_spectral")
            axes[4].set_title("5. Connected Components")
            axes[4].axis("off")

        # Touches significatives
        key_regions = [r for r in regions if (r.area > 1000 or r.area < 2000)]
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
        if debug == True:
            # Display key regions in axis 5
            axes[5].imshow(labels, cmap="nipy_spectral")
            axes[5].set_title(f"6. Key Regions (Count: {len(key_regions)})")
            axes[5].axis('off')
            print(key_regions)
            # Display centroids in axis 6
            axes[6].imshow(labels, cmap="nipy_spectral")
            axes[6].set_title(f"7. Centroids (Count: {len(centroids)})")
            axes[6].axis('off')
            print(centroids)
            # Plot centroids on axis 6
            for centroid in centroids:
                axes[6].plot(centroid[1], centroid[0], 'ro', markersize=4)
        plt.tight_layout()
        plt.show()

        if diffs[0] > np.mean(diffs) * 1.3:
            results.append("QWERTY") #needs to be replaced with a sortqwertyfunction (us/uk etc.)
        elif diffs[0] < np.mean(diffs) * 0.7:
            results.append("AZERTY") #needs to be replaced with a sortawertyfunction (belgium/france etc. )
        else:
            results.append(describe_unknown_keyboard([img_path]))
    return results


def describe_unknown_keyboard(img_paths):
    """Lightweight keyboard layout detection using OCR + simple pattern matching."""

    patterns = {
        'QWERTY': list("QWERTYUIOP"),
        'AZERTY': list("AZERTYUIOP"),
        'DVORAK': list("AOEUIDHTNS"),
        'COLEMAK': list("QWFGJ KLSM".replace(" ", ""))
    }

    results = []
