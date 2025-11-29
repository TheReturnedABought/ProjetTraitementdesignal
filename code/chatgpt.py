import skimage
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pytesseract


def detect_layout_from_image(img_paths, debug=False):

    results = []

    for img_path in img_paths:
        img = plt.imread(img_path)
        if img.shape[-1] == 4:
            img = img[..., :3]

        # 1. Convert grayscale
        gray = skimage.color.rgb2gray(img)

        # 2. Threshold
        thresh = skimage.filters.threshold_otsu(gray)
        binary = gray < thresh

        # 3. Remove small noise
        binary = skimage.morphology.remove_small_objects(binary, min_size=150)

        # 4. Label objects
        labels = skimage.measure.label(binary)
        regions = skimage.measure.regionprops(labels)

        # 5. Filter real keys
        key_regions = [
            r for r in regions
            if 800 < r.area < 8000
            and r.solidity > 0.45
            and r.eccentricity < 0.995
        ]

        if len(key_regions) < 20:
            results.append("Unknown")
            continue

        # 6. Centroids
        centroids = np.array([r.centroid for r in key_regions])

        # 7. Sort by y (top to bottom)
        centroids_sorted = centroids[np.argsort(centroids[:, 0])]

        # 8. First row estimation (keys closest in Y)
        first_row = centroids_sorted[:15]

        # cluster based on Y spacing
        y_vals = first_row[:, 0]
        median_y = np.median(y_vals)
        tolerance = 15
        first_row = centroids[np.abs(centroids[:, 0] - median_y) < tolerance]

        # sort first row left → right
        first_row = first_row[np.argsort(first_row[:, 1])]

        # 9. Crop the first-left key (should be Q or A)
        y, x = first_row[0]
        y, x = int(y), int(x)

        box = 30  # crop around centroid
        crop = img[max(0, y-box): y+box, max(0, x-box): x+box]

        # OCR on the cropped key
        try:
            letter = pytesseract.image_to_string(
                Image.fromarray((crop*255).astype(np.uint8)),
                config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ).strip().upper()
        except:
            letter = ""

        # 10. Decision
        if letter == "A":
            results.append("AZERTY")
        elif letter == "Q":
            results.append("QWERTY")
        else:
            # fallback: use geometry
            xs = first_row[:, 1]
            diffs = np.diff(xs)

            if diffs[0] > np.mean(diffs)*1.3:
                results.append("QWERTY")
            elif diffs[0] < np.mean(diffs)*0.7:
                results.append("AZERTY")
            else:
                results.append("Unknown")

        if debug:
            print("OCR detected:", letter)
            plt.imshow(crop)
            plt.title("Leftmost key cropped")
            plt.show()

    return results
