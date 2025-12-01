import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from collections import Counter
from typing import List, Tuple, Dict, Any
import warnings

from sklearn.utils.multiclass import type_of_target

from utils import *

warnings.filterwarnings('ignore')

# =============================
# LAYOUT DEFINITIONS
# =============================

LAYOUTS = {
    "QWERTY": {
        "row1": {
            "chars": list("QWERTYUIOP"),
            "indicators": ["Q", "W", "Y"],
            "conflicts": ["Ç"]
        },
        "row2": {
            "chars": list("ASDFGHJKL"),
            "indicators": ["A", "S"],
            "conflicts": []
        },
        "row3": {
            "chars": list("ZXCVBNM"),
            "indicators": ["Z"],
            "conflicts": ["É"]
        }
    },
    "AZERTY": {
        "row1": {
            "chars": list("AZERTYUIOP"),
            "indicators": ["A", "Z", "E"],
            "conflicts": ["W"]
        },
        "row2": {
            "chars": list("QSDFGHJKLM"),
            "indicators": ["Q", "S"],
            "conflicts": []
        },
        "row3": {
            "chars": list("WXCVBN"),
            "indicators": ["X"],
            "conflicts": ["Y"]
        }
    },
    "QWERTZ": {
        "row1": {
            "chars": list("QWERTZUIOP"),
            "indicators": ["Q", "W", "Z"],
            "conflicts": ["Y"]
        },
        "row2": {
            "chars": list("ASDFGHJKL"),
            "indicators": ["A", "S"],
            "conflicts": []
        },
        "row3": {
            "chars": list("YXCVBNM"),
            "indicators": ["Y"],
            "conflicts": ["Z"]
        }
    }
}

# Character mapping for OCR corrections
LIKELY_KEYMAP = {
    # Single-char digit/letter confusions
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "W",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "G",
    # Multi-char confusions
    "41": "Q",
    "I3": "B",
    "13": "B",
    "E8": "B",
    "00": "O",
    "0Q": "Q",
    "OQ": "Q",
    "QQ": "Q",
    "AA": "A",
    "ZZ": "Z",
    "WW": "W",
    "EE": "E",
    "RR": "R",
    "TT": "T",
    "YY": "Y",
    "UU": "U",
    "II": "I",
    "OO": "O",
    "PP": "P",
}

# =============================
# PREPROCESSING METHODS
# =============================

def method1_contrast_and_sharpen(img: np.ndarray) -> np.ndarray:
    img = increase_contrast(img)
    img = sharpen_image(img)
    img = convert_to_gray(img)
    return img

def method2_blur_and_sharpen(img):
    img = apply_gaussian_blur(img)
    img = sharpen_image(img)
    img = convert_to_gray(img)
    return img

def method3_simple_inversion(img: np.ndarray) -> np.ndarray:
    """Simple grayscale inversion."""
    """Simple grayscale inversion."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    return cv2.bitwise_not(gray)

def method4_upscaled_contrast_blur_and_sharpen(img: np.ndarray) -> np.ndarray:
    img = apply_gaussian_blur(img)
    img = sharpen_image(img)
    img = sharpen_image(img)
    img = convert_to_gray(img)
    return img

# =============================
# OCR CHARACTER MAPPING
# =============================
def _map_likely(text: str) -> str:
    """Map a possibly-misread string to a single character using LIKELY_KEYMAP."""
    if not text:
        return text

    t = text.strip().upper()

    # Direct mapping
    if t in LIKELY_KEYMAP:
        return LIKELY_KEYMAP[t]

    # Multi-char to single char mapping
    if len(t) > 1 and t in LIKELY_KEYMAP:
        return LIKELY_KEYMAP[t]

    # Try to extract single alpha character
    alpha = "".join([c for c in t if c.isalpha()])
    if len(alpha) == 1:
        return alpha

    # Fallback: take first character and map if possible
    if t and t[0] in LIKELY_KEYMAP:
        return LIKELY_KEYMAP[t[0]]

    return t[0] if t else ""
# =============================
# OCR DETECTION
# =============================
def ocr_keyboard_layout(reader: easyocr.Reader, processed_images: List[np.ndarray]) -> Tuple:
    """
    Perform OCR on multiple preprocessed keyboard images and aggregate character detections.

    This function applies EasyOCR to a list of processed images, extracts detected
    characters, filters low-confidence results, applies likely character mappings
    to correct OCR misreads, and aggregates results across different preprocessing methods.

    Steps:
        1. OCR is run on each processed image using an allowlist of uppercase letters and digits.
        2. Low-confidence detections (confidence < 0.5) and long strings (>=3 chars) are ignored.
        3. Detected text is normalized to uppercase and mapped using `_map_likely` to correct common OCR errors.
        4. Characters detected by at least 2 methods are considered "validated".
        5. Aggregates all unique characters, counts per character, and detections per method.

    Args:
        reader (easyocr.Reader): Initialized EasyOCR reader object.
        processed_images (List[np.ndarray]): List of preprocessed images (grayscale or binary) ready for OCR.

    Returns:
        Tuple:
            validated (List[str]): Characters detected in at least 2 methods.
            all_unique (List[str]): All unique characters detected across all methods.
            char_counts (Counter): Count of each detected character across all methods.
            all_detected (List[List[Tuple[str, float]]]): List of detected characters per method with confidence.
            method_names (List[str]): Names of the preprocessing methods corresponding to each detection list.
    """
    all_detected = []
    method_names = ["Adaptive Threshold", "LAB Channel", "Simple Inversion", "Text"]

    for proc_img in processed_images:
        results = reader.readtext(
            proc_img,
            detail=1,
            allowlist='AZERTYUIOPQSDFGHJKLMWXCVBN0123456789',
            text_threshold=0.4,
            low_text=0.2,
            link_threshold=0.2,
        )

        detected = []
        for (bbox, text, conf) in results:
            text = text.strip().upper()
            if conf is None:
                conf = 0.0

            # Filter out long strings and low confidence
            if len(text) >= 3:
                continue
            if conf < 0.50:
                continue

            # Apply character mapping
            if len(text) != 1:
                text = _map_likely(text)
            if text and not ('A' <= text <= 'Z'):
                text = _map_likely(text)

            if text and 'A' <= text <= 'Z':
                detected.append((text, conf))

        all_detected.append(detected)

    # Combine chars from all methods
    all_chars = []
    for det in all_detected:
        for t, _ in det:
            if len(t) == 1:
                all_chars.append(t)

    char_counts = Counter(all_chars)

    # Validation: require at least 2 detections across methods
    validated = [c for c, count in char_counts.items() if count >= 2]
    all_unique = list(set(all_chars))

    return validated, all_unique, char_counts, all_detected, method_names
# =============================
# ROW CLUSTERING & SCORING
# =============================
def assign_rows_to_chars(detected_letters: List[str], cluster_labels: Dict[str, int]) -> Dict[int, List[str]]:
    """
    Organize detected characters into keyboard rows based on cluster assignments.

    This function groups each character from `detected_letters` into its corresponding
    keyboard row (top, middle, bottom) using the provided `cluster_labels`. The output
    makes it easy to analyze or score the layout by row.

    Args:
        detected_letters (List[str]): List of uppercase characters detected from OCR.
        cluster_labels (Dict[str, int]): Mapping of characters to row indices:
                                         0 = top row, 1 = middle row, 2 = bottom row.

    Returns:
        Dict[int, List[str]]: Dictionary with row indices as keys (0, 1, 2) and lists
                              of characters assigned to each row as values.
    """
    row_chars = {0: [], 1: [], 2: []}  # 0=top, 1=mid, 2=bottom

    for char in detected_letters:
        if char in cluster_labels:
            row = cluster_labels[char]
            row_chars[row].append(char)
        print(char, ": ", row_chars[row])
    return row_chars
# =============================
# VISUALIZATION
# =============================
def visualize_results(
        img_original: np.ndarray,
        processed_images: List[np.ndarray],
        ocr_results: List,
        ocr_full_detections: List,
        method_names: List[str],
        layout_result: str = "",
        confidence: float = 0.0,
        detected_chars: List[str] = None
):
    if detected_chars is None:
        detected_chars = []

    # Calculer le nombre de colonnes nécessaires (original + processed)
    num_cols = len(processed_images) + 1
    fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 10))

    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Summary panel (en bas à gauche)
    text_canvas = np.ones((400, 600, 3), dtype=np.uint8) * 255
    y = 40
    cv2.putText(text_canvas, "DETECTION SUMMARY", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    y += 60
    cv2.putText(text_canvas, f"Detected Layout: {layout_result}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 0), 2)
    y += 50
    cv2.putText(text_canvas, f"Confidence: {confidence:.1f}%", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    y += 50
    cv2.putText(text_canvas, f"Detected Chars ({len(detected_chars)}):", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    y += 40
    chars_text = " ".join(sorted(set(detected_chars)))
    cv2.putText(text_canvas, chars_text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 200), 2)

    axes[1, 0].imshow(text_canvas)
    axes[1, 0].set_title("Layout Detection")
    axes[1, 0].axis("off")

    # Chaque méthode de prétraitement
    for i, (proc_img, name, detections, full_detections) in enumerate(
            zip(processed_images, method_names, ocr_results, ocr_full_detections)
    ):
        # Dessiner les bounding boxes sur l'image
        img_with_boxes = draw_boxes_on_image(proc_img, full_detections)

        axes[0, i + 1].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        axes[0, i + 1].set_title(f"{name}\n({len(detections)} chars)")
        axes[0, i + 1].axis("off")

        # OCR text panel
        text_canvas = np.ones((400, 600, 3), dtype=np.uint8) * 255
        y = 30
        for t, conf in detections:
            cv2.putText(
                text_canvas, f"{t} ({conf * 100:.1f}%)",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 2
            )
            y += 35
            if y > 370:  # Éviter de dépasser le canvas
                break

        axes[1, i + 1].imshow(text_canvas)
        axes[1, i + 1].set_title(f"OCR Output")
        axes[1, i + 1].axis("off")

    plt.tight_layout()
    plt.show()

def score_layout_with_row_clustering(detected_letters: List[str], cluster_labels: Dict[str, int]) -> Dict[str, float]:
    """
    Improved scoring that focuses on layout DIFFERENCES.
    """
    scores = {}
    detected_set = set(detected_letters)

    # Define layout-specific patterns that are KEY differentiators
    layout_patterns = {
        'AZERTY': {
            'top_row_chars': set('AZERTYUIOP'),
            'mid_row_chars': set('QSDFGHJKLM'),
            'bottom_row_chars': set('WXCVBN'),
            'key_indicators': {'A', 'Z', 'Q', 'W', 'M'},  # A,Z top; Q middle; W,M bottom in AZERTY
            'conflict_chars': {'Y'},  # Y is top row in QWERTY but not in AZERTY
            'unique_to_layout': {'M'},  # M is middle row in AZERTY, bottom in others
        },
        'QWERTY': {
            'top_row_chars': set('QWERTYUIOP'),
            'mid_row_chars': set('ASDFGHJKL'),
            'bottom_row_chars': set('ZXCVBNM'),
            'key_indicators': {'Q', 'W', 'Y', 'A', 'S'},
            'conflict_chars': set(),
            'unique_to_layout': set(),
        },
        'QWERTZ': {
            'top_row_chars': set('QWERTZUIOP'),
            'mid_row_chars': set('ASDFGHJKL'),
            'bottom_row_chars': set('YXCVBNM'),
            'key_indicators': {'Q', 'W', 'Z', 'Y'},
            'conflict_chars': set(),
            'unique_to_layout': set(),
        }
    }

    for layout_name, pattern in layout_patterns.items():
        score = 0.0
        layout_info = LAYOUTS[layout_name]
        rows = [layout_info['row1'], layout_info['row2'], layout_info['row3']]

        # 1. Row correctness score (if we have row assignments)
        row_correctness = 0
        if cluster_labels:
            for char in detected_letters:
                if char in cluster_labels:
                    actual_row = cluster_labels[char]
                    expected_row = None

                    # Find which row this character should be in
                    for i, row_info in enumerate(rows):
                        if char in row_info['chars']:
                            expected_row = i
                            break

                    if expected_row is not None:
                        if actual_row == expected_row:
                            # Perfect match!
                            row_correctness += 3.0
                        else:
                            # Wrong row
                            row_correctness -= 0.2
                    else:
                        # Character not in this layout
                        row_correctness -= 1.0

        # 2. Character presence score (weighted by importance)
        char_presence = 0
        for char in detected_letters:
            in_top = char in pattern['top_row_chars']
            in_mid = char in pattern['mid_row_chars']
            in_bottom = char in pattern['bottom_row_chars']

            if in_top or in_mid or in_bottom:
                # Character exists in layout
                if char in pattern['key_indicators']:
                    char_presence += 2.5  # Key indicator bonus
                elif in_top:
                    char_presence += 1.2  # Top row more important
                elif in_mid:
                    char_presence += 1.0  # Middle row
                else:
                    char_presence += 0.8  # Bottom row

        # 3. Layout-specific bonus/penalty
        layout_specific = 0

        # Bonus for characters unique to this layout
        for unique_char in pattern['unique_to_layout']:
            if unique_char in detected_set:
                layout_specific += 5.0

        # Penalty for conflict characters
        for conflict_char in pattern['conflict_chars']:
            if conflict_char in detected_set:
                layout_specific -= 4.0

        # 4. Row distribution bonus (encourage balanced detection across rows)
        top_matches = detected_set & pattern['top_row_chars']
        mid_matches = detected_set & pattern['mid_row_chars']
        bottom_matches = detected_set & pattern['bottom_row_chars']

        row_distribution = min(len(top_matches), len(mid_matches), len(bottom_matches))
        distribution_bonus = row_distribution * 1.5

        # 5. Layout coherence check
        coherence = 0
        # Check if key AZERTY indicators are consistent
        if layout_name == 'AZERTY':
            azerty_consistent = True
            # A and Z should be in top row if we have row info
            if cluster_labels:
                if 'A' in cluster_labels and 'Z' in cluster_labels:
                    if cluster_labels['A'] != 0 or cluster_labels['Z'] != 0:
                        azerty_consistent = False
                # W should be in bottom row
                if 'W' in cluster_labels:
                    if cluster_labels['W'] != 2:
                        azerty_consistent = False

            if azerty_consistent:
                coherence += 5.0

        if layout_name == 'QWERTY':
            qwerty_consistent = True
            # Q and W should be in top row if we have row info
            if cluster_labels:
                if 'Q' in cluster_labels and 'W' in cluster_labels:
                    if cluster_labels['Q'] != 0 or cluster_labels['W'] != 0:
                        qwerty_consistent = False
                # W should be in bottom row
                if 'Z' in cluster_labels:
                    if cluster_labels['Z'] != 2:
                        qwerty_consistent = False

            if qwerty_consistent:
                coherence += 5.0

        # Total score with weights
        total_score = (
                row_correctness * 0.4 +
                char_presence * 0.3 +
                layout_specific * 0.2 +
                distribution_bonus * 0.05 +
                coherence * 0.05
        )

        scores[layout_name] = max(0, total_score)

    return scores
def get_row_labels_for_validated_chars(reader, processed_images, validated_chars):
    """
    Get row assignments for VALIDATED characters only.
    Uses keyboard row spacing heuristics.
    """
    char_positions = {}

    # For each preprocessing method
    for proc_img in processed_images:
        raw_results = reader.readtext(
            proc_img,
            detail=1,
            allowlist='AZERTYUIOPQSDFGHJKLMWXCVBN0123456789',
            text_threshold=0.4,
            low_text=0.2,
            link_threshold=0.2,
        )

        for (bbox, text, conf) in raw_results:
            if not bbox or not text or conf < 0.5:
                continue

            text = text.strip().upper()

            # Apply character mapping
            if len(text) != 1:
                text = _map_likely(text)

            if not text or not ('A' <= text <= 'Z') or text not in validated_chars:
                continue

            # Get y-center of bounding box
            y_coords = [point[1] for point in bbox]
            y_center = sum(y_coords) / len(y_coords)

            # Store weighted average position
            if text not in char_positions:
                char_positions[text] = {'total_y': 0, 'total_conf': 0, 'count': 0}

            char_positions[text]['total_y'] += y_center * conf
            char_positions[text]['total_conf'] += conf
            char_positions[text]['count'] += 1

    if len(char_positions) < 3:
        return {}

    # Calculate weighted average y-position for each character
    char_avg_y = {}
    for char, data in char_positions.items():
        if data['total_conf'] > 0:
            weighted_y = data['total_y'] / data['total_conf']
            char_avg_y[char] = weighted_y

    # Sort characters by y-position
    sorted_chars = sorted(char_avg_y.items(), key=lambda x: x[1])

    # Extract y-values for clustering
    y_values = np.array([y for _, y in sorted_chars]).reshape(-1, 1)

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(y_values)

    # Predict row clusters
    clusters = gmm.predict(y_values)

    # Get cluster means and sort them (top to bottom)
    cluster_means = gmm.means_.flatten()
    sorted_clusters = np.argsort(cluster_means)

    # Map clusters to row numbers
    cluster_to_row = {sorted_clusters[i]: i for i in range(3)}
    print("Rows for each Char: ")
    # Create final mapping
    row_assignments = {}
    for (char, _), cluster in zip(sorted_chars, clusters):
        row_assignments[char] = cluster_to_row.get(cluster, 1)
        print(char, ": ", row_assignments[char])
    return row_assignments
# =============================
# MAIN DETECTION FUNCTION
# =============================
def detect_layout_from_image(
        img_path,
        use_row_clustering: bool = True,
        debug: bool = True
) -> Dict[str, Tuple[str, float, List[str], Dict]]:
    """
    Main function to detect keyboard layout from one or multiple images.
    """
    if isinstance(img_path, str):
        img_paths = [img_path]
    elif isinstance(img_path, list):
        img_paths = img_path
    else:
        raise TypeError("img_path must be a string or list of strings")

    results = {}

    for path in img_paths:
        print(f"\n{'=' * 60}\nProcessing: {path}\n{'=' * 60}")

        # Load image
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not load image: {path}")
            continue
        img = upscale_image(img)
        # Apply preprocessing methods
        processed_images = [
            method1_contrast_and_sharpen(img),
            method2_blur_and_sharpen(img),
            method3_simple_inversion(img),
            method4_upscaled_contrast_blur_and_sharpen(img)
        ]

        # Initialize OCR reader
        reader = easyocr.Reader(['en', 'fr'], gpu=False)

        # Get OCR results
        validated, all_unique, char_counts, all_detected, method_names = ocr_keyboard_layout(
            reader, processed_images
        )

        # Use validated characters if available, otherwise all unique
        detected_chars = validated if validated else all_unique
        detected_chars_set = set(detected_chars) if isinstance(detected_chars, list) else detected_chars

        # Debug output
        print(f"\nDetected characters: {sorted(detected_chars)}")
        print(f"Character counts: {char_counts}")

        # Get row assignments using ONLY validated characters
        row_labels = {}
        if use_row_clustering and detected_chars:
            row_labels = get_row_labels_for_validated_chars(reader, processed_images, detected_chars)
            print(f"Row assignments for validated chars: {row_labels}")

        # Score layouts
        scores = {}
        if row_labels and len(row_labels) >= 3:
            scores = score_layout_with_row_clustering(detected_chars_set, row_labels)  # Use set here
        else:
            # Fall back to basic scoring without row info
            scores = score_layout_with_row_clustering(detected_chars_set, {})  # Use set here

        # Choose best layout
        if scores:
            best_layout = max(scores, key=scores.get)

            # Calculate confidence based on:
            # 1. Score relative to other layouts
            # 2. Presence of key indicators
            # 3. Number of characters detected

            max_score = scores[best_layout]
            second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0

            # Score difference
            score_diff = max_score - second_best

            # Base confidence
            if max_score > 0:
                base_confidence = min(100, (max_score / 30.0) * 100.0)
            else:
                base_confidence = 0

            # Boost confidence if we have key indicators
            layout_indicators = {
                'AZERTY': {'A', 'Z', 'Q', 'W', 'M'},
                'QWERTY': {'Q', 'W', 'Y', 'A', 'S'},
                'QWERTZ': {'Q', 'W', 'Z', 'Y'}
            }
            confidence = base_confidence
        else:
            best_layout = "Unknown"
            confidence = 0.0

        print(f"\nDetection Results for {path}:")
        print(f"  Layout: {best_layout}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Final scores: {scores}")

        # Debug: Show layout analysis
        if debug:
            print("\nLayout Analysis:")
            for layout_name in LAYOUTS:
                layout_info = LAYOUTS[layout_name]
                top_row = set(layout_info['row1']['chars'])
                mid_row = set(layout_info['row2']['chars'])
                bottom_row = set(layout_info['row3']['chars'])

                top_matches = detected_chars_set & top_row  # Now using set
                mid_matches = detected_chars_set & mid_row  # Now using set
                bottom_matches = detected_chars_set & bottom_row  # Now using set

                print(f"  {layout_name}:")
                print(f"    Top row matches: {sorted(top_matches)}")
                print(f"    Middle row matches: {sorted(mid_matches)}")
                print(f"    Bottom row matches: {sorted(bottom_matches)}")

        # Visualization
        if debug:
            visualize_results(
                img, processed_images, all_detected,
                method_names, best_layout, confidence, detected_chars  # Keep original for visualization if needed
            )

        results[path] = (best_layout, confidence)

    return results