#trie
import skimage
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure
import skimage.morphology
from skimage.filters import threshold_local
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pytesseract
from utils import sobel

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Théo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


def detect_layout_from_image(img_paths, debug=False):

    results = []

    for img_path in img_paths:

        fig, axes = plt.subplots(1, 10, figsize=(32, 6))

        img = plt.imread(img_path)
        if img.shape[-1] == 4:
            img = img[..., :3]   # remove alpha

        # 1. IMAGE ORIGINALE
        if debug:
            axes[0].imshow(img)
            axes[0].set_title("1. Original")
            axes[0].axis("off")

        # 2. GRAYSCALE
        gray = skimage.color.rgb2gray(img)
        if debug:
            axes[1].imshow(gray, cmap="gray")
            axes[1].set_title("2. Gray")
            axes[1].axis("off")

        # 3. SOBEL
        edges = sobel(gray)
        if debug:
            axes[2].imshow(edges, cmap="gray")
            axes[2].set_title("3. Sobel")
            axes[2].axis("off")

        # 4. BINARY — ADAPTATIVE THRESHOLD (IMPORTANT)
        T = threshold_local(edges, block_size=51, offset=5)
        binary = edges > T

        # cleanup
        binary = skimage.morphology.remove_small_objects(binary, min_size=50)
        binary = skimage.morphology.remove_small_holes(binary, area_threshold=200)

        if debug:
            axes[3].imshow(binary, cmap="gray")
            axes[3].set_title("4. Binary (Adaptive)")
            axes[3].axis("off")

        # 5. LABELS
        labels = skimage.measure.label(binary)
        regions = skimage.measure.regionprops(labels)
        if debug:
            axes[4].imshow(labels, cmap="nipy_spectral")
            axes[4].set_title("5. Labels")
            axes[4].axis("off")

        # 6. KEY REGIONS – automatic size-based filtering
        H, W = edges.shape
        area_min = (H * W) * 0.00010   # 0.01%
        area_max = (H * W) * 0.004     # 0.4%

        key_regions = [r for r in regions if area_min < r.area < area_max]

        if debug:
            axes[5].imshow(labels, cmap="nipy_spectral")
            axes[5].set_title(f"6. Key Regions ({len(key_regions)})")
            axes[5].axis("off")

        if len(key_regions) < 10:
            results.append("Unknown")
            if debug:
                plt.tight_layout(); plt.show()
            continue

        # 7. CENTROIDS
        centroids = np.array([r.centroid for r in key_regions])
        if debug:
            axes[6].imshow(labels, cmap="nipy_spectral")
            for c in centroids:
                axes[6].plot(c[1], c[0], "ro", markersize=3)
            axes[6].set_title("7. Centroids")
            axes[6].axis("off")

        # --- END DEBUG ---
        if debug:
            axes[9].text(0.2, 0.5, "Debug done", fontsize=16)
            axes[9].axis("off")
            plt.tight_layout()
            plt.show()

        # --- DECISION (geometry-based) ---

        # Sort by Y → first/upper row
        centroids_sorted = centroids[np.argsort(centroids[:, 0])]
        first_row = centroids_sorted[:12]
        first_row = first_row[np.argsort(first_row[:, 1])]

        xs = first_row[:, 1]
        diffs = np.diff(xs)

        if len(diffs) < 2:
            results.append("Unknown")
            continue

        gap0 = diffs[0]
        mean_gap = np.mean(diffs[1:])

        if gap0 < mean_gap * 0.7:
            results.append("QWERTY")
        elif gap0 > mean_gap * 1.3:
            results.append("AZERTY")
        else:
            results.append(describe_unknown_keyboard([img_path]))

    return results



# -------------------------------------------------------------
# OCR fallback
# -------------------------------------------------------------
def describe_unknown_keyboard(img_paths):

    patterns = {
        'QWERTY': "QWERTYUIOP",
        'AZERTY': "AZERTYUIOP",
        'QWERTZ': "QWERTZUIOP",
        'DVORAK': "AOEUIDHTNS",
        'COLEMAK': "QWFPGJLUY"
    }

    results = []

    for img_path in img_paths:

        img = Image.open(img_path).convert("L")

        img_bin = img.point(lambda x: 0 if x < 120 else 255, "1")

        raw_text = pytesseract.image_to_string(
            img_bin,
            config="--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

        text = "".join([c for c in raw_text.upper() if c.isalpha()])

        print("OCR full image:", text)

        if len(text) < 3:
            results.append("Unknown")
            continue

        best_layout = "Unknown"
        best_score = -1

        for layout, pattern in patterns.items():
            score = sum(c in text for c in pattern)
            if score > best_score:
                best_score = score
                best_layout = layout

        results.append(best_layout)

    return results
