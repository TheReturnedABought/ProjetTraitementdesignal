import numpy as np
from skimage import color, filters, measure, morphology

def detect_layout_from_image(img):
    """
    Detect AZERTY vs QWERTY using ONLY signal-processing tools.
    Input:
        img  -> an RGB image loaded with skimage.io.imread
    Output:
        "AZERTY", "QWERTY", or "Unknown"
    """

    # --- 1. Convert to grayscale ---
    gray = color.rgb2gray(img)

    # --- 2. Threshold (binary image) ---
    th = filters.threshold_otsu(gray)
    binary = gray < th  # depending on contrast, you may invert (< or >)

    # Clean binary image (remove small noise)
    binary = morphology.remove_small_objects(binary, min_size=300)

    # --- 3. Label connected components ---
    labels = measure.label(binary)
    regions = measure.regionprops(labels)

    # Keep only large objects (keys)
    key_regions = [r for r in regions if r.area > 1000]

    if len(key_regions) < 10:   # safety check
        return "Unknown"

    # --- 4. Extract centroids (x = column, y = row) ---
    centroids = np.array([r.centroid for r in key_regions])

    # --- 5. Select first-row keys (smallest y = top of the keyboard) ---
    # Sort by vertical (row) coordinate
    centroids_sorted = centroids[centroids[:, 0].argsort()]

    # Take the first ~10 keys (enough for one row)
    first_row = centroids_sorted[:10]

    # Sort them horizontally (by x column position)
    first_row = first_row[first_row[:, 1].argsort()]

    # Compute horizontal distances between adjacent keys (spacing)
    xs = first_row[:, 1]
    diffs = np.diff(xs)

    # Heuristic:
    # On QWERTY, the first key (Q) is shifted more right -> first spacing bigger
    # On AZERTY, the first key (A) sits more left -> spacing smaller
    if diffs[0] > np.mean(diffs) * 1.3:
        return "QWERTY"
    if diffs[0] < np.mean(diffs) * 0.7:
        return "AZERTY"

    return "Unknown"
