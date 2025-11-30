import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings

warnings.filterwarnings('ignore')


def detect_layout_from_image(img_paths, debug=False):
    """
    Enhanced keyboard layout detection using signal processing and image analysis
    """
    results = []

    for img_path in img_paths:
        try:
            # ==========================================
            # 1. IMAGE PRE-PROCESSING (Syllabus Part 4)
            # ==========================================
            img = cv2.imread(img_path)
            if img is None:
                results.append("Error: File not found")
                continue

            # Convert to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize for consistent processing
            h, w = img.shape[:2]
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ==========================================
            # 2. FREQUENCY DOMAIN ANALYSIS (Syllabus Part 3)
            # ==========================================
            # Apply FFT to detect periodic patterns (keyboard rows)
            fft_result = np.fft.fft2(gray.astype(float))
            fft_shift = np.fft.fftshift(fft_result)
            magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)

            # ==========================================
            # 3. ADAPTIVE THRESHOLDING WITH MORPHOLOGY
            # ==========================================
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 51, 5
            )

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=3)

            # ==========================================
            # 4. EDGE DETECTION (Syllabus: Sobel filters)
            # ==========================================
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # ==========================================
            # 5. CONTOUR DETECTION AND ANALYSIS
            # ==========================================
            contours, _ = cv2.findContours(
                processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            key_centroids = []
            key_areas = []
            debug_img = img_rgb.copy()

            # Filter contours based on area and aspect ratio
            for c in contours:
                area = cv2.contourArea(c)
                if area < 100:  # Minimum area threshold
                    continue

                x, y, ww, hh = cv2.boundingRect(c)
                aspect_ratio = float(ww) / hh if hh > 0 else 0

                # Key-like aspect ratios
                if 0.3 < aspect_ratio < 3.0:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        key_centroids.append([cX, cY])
                        key_areas.append(area)

                        if debug:
                            cv2.drawContours(debug_img, [c], -1, (0, 255, 0), 2)
                            cv2.circle(debug_img, (cX, cY), 4, (255, 0, 0), -1)

            if len(key_centroids) < 10:
                results.append("Unknown")
                continue

            key_centroids = np.array(key_centroids)
            key_areas = np.array(key_areas)

            # ==========================================
            # 6. ROW DETECTION WITHOUT MACHINE LEARNING
            # ==========================================
            # Sort by Y coordinate and group into rows using histogram-based approach
            y_coords = key_centroids[:, 1]

            # Use histogram to find row clusters in Y-axis
            hist, bin_edges = np.histogram(y_coords, bins=min(10, len(key_centroids) // 3))
            peak_bins = signal.find_peaks(hist, height=2)[0]

            rows = []
            if len(peak_bins) > 0:
                # Group keys by Y proximity to histogram peaks
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                row_centers = bin_centers[peak_bins]

                for center in row_centers:
                    # Find keys close to this row center
                    mask = np.abs(y_coords - center) < (bin_edges[1] - bin_edges[0]) * 1.5
                    row_keys = key_centroids[mask]
                    if len(row_keys) >= 5:  # Minimum keys per row
                        # Sort by X coordinate
                        row_keys = row_keys[row_keys[:, 0].argsort()]
                        rows.append(row_keys)

            # If histogram method fails, use simple sorting and grouping
            if len(rows) < 2:
                rows = []
                sorted_indices = np.argsort(y_coords)
                sorted_centroids = key_centroids[sorted_indices]

                current_row = [sorted_centroids[0]]
                y_tolerance = np.median(key_areas) ** 0.5 * 0.6

                for i in range(1, len(sorted_centroids)):
                    if abs(sorted_centroids[i][1] - current_row[-1][1]) < y_tolerance:
                        current_row.append(sorted_centroids[i])
                    else:
                        if len(current_row) >= 5:
                            current_row = sorted(current_row, key=lambda x: x[0])
                            rows.append(np.array(current_row))
                        current_row = [sorted_centroids[i]]

                if len(current_row) >= 5:
                    current_row = sorted(current_row, key=lambda x: x[0])
                    rows.append(np.array(current_row))

            # Sort rows by Y coordinate
            rows.sort(key=lambda r: np.mean(r[:, 1]))

            # ==========================================
            # 7. PATTERN ANALYSIS USING SIGNAL PROCESSING
            # ==========================================
            decision = "Unknown"
            confidence = 0

            if len(rows) >= 2:
                # Analyze the home row (usually second row)
                target_row_idx = 1 if len(rows) > 2 else 0
                target_row = rows[target_row_idx]

                if len(target_row) >= 8:
                    # Extract spacing pattern as signal
                    x_coords = target_row[:, 0]
                    spacings = np.diff(x_coords)

                    if len(spacings) >= 4:
                        # Normalize spacings
                        normalized_spacings = spacings / np.mean(spacings)

                        # Calculate statistical features
                        first_spacing_ratio = normalized_spacings[0]
                        spacing_std = np.std(normalized_spacings[:min(6, len(normalized_spacings))])

                        # Apply Fourier analysis to spacing pattern
                        spacing_fft = np.fft.fft(normalized_spacings)
                        fft_magnitude = np.abs(spacing_fft)

                        # Look for dominant frequencies (pattern regularity)
                        dominant_freq = np.argmax(fft_magnitude[1:len(fft_magnitude) // 2]) + 1

                        # Decision logic based on signal characteristics
                        azerty_features = 0
                        qwerty_features = 0

                        # Feature 1: First spacing (AZERTY often has smaller first gap)
                        if first_spacing_ratio < 0.85:
                            azerty_features += 2
                        elif first_spacing_ratio > 0.95:
                            qwerty_features += 1

                        # Feature 2: Spacing uniformity (QWERTY is more uniform)
                        if spacing_std < 0.15:
                            qwerty_features += 2
                        else:
                            azerty_features += 1

                        # Feature 3: Pattern regularity from FFT
                        if dominant_freq <= 2:  # Low frequency dominant = more regular
                            qwerty_features += 1
                        else:
                            azerty_features += 1

                        # Make final decision
                        if azerty_features > qwerty_features:
                            decision = "AZERTY"
                            confidence = azerty_features
                        else:
                            decision = "QWERTY"
                            confidence = qwerty_features

            # ==========================================
            # 8. COMPREHENSIVE DEBUG VISUALIZATION
            # ==========================================
            if debug:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))

                # 1. Original image
                axes[0, 0].imshow(img_rgb)
                axes[0, 0].set_title("1. Original Image")
                axes[0, 0].axis('off')

                # 2. Frequency spectrum
                axes[0, 1].imshow(magnitude_spectrum, cmap='hot')
                axes[0, 1].set_title("2. Frequency Spectrum (FFT)")
                axes[0, 1].axis('off')

                # 3. Edge detection
                axes[0, 2].imshow(sobel_magnitude, cmap='gray')
                axes[0, 2].set_title("3. Sobel Edge Detection")
                axes[0, 2].axis('off')

                # 4. Binary processing
                axes[1, 0].imshow(processed, cmap='gray')
                axes[1, 0].set_title("4. Binary Processing")
                axes[1, 0].axis('off')

                # 5. Detected keys with contours
                axes[1, 1].imshow(debug_img)
                axes[1, 1].set_title(f"5. Detected Keys ({len(key_centroids)})")
                axes[1, 1].axis('off')

                # 6. Row detection results
                row_vis = img_rgb.copy()
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

                for i, row in enumerate(rows):
                    color = colors[i % len(colors)]
                    for pt in row:
                        cv2.circle(row_vis, (int(pt[0]), int(pt[1])), 8, color, -1)

                    # Draw row number and highlight target row
                    if len(row) > 0:
                        mean_y = np.mean(row[:, 1])
                        row_text = f'Row {i}'
                        if i == (1 if len(rows) > 2 else 0):
                            row_text += ' (TARGET)'
                            # Highlight target row with white border
                            for pt in row:
                                cv2.circle(row_vis, (int(pt[0]), int(pt[1])), 10, (255, 255, 255), 2)

                        cv2.putText(row_vis, row_text,
                                    (10, int(mean_y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                axes[1, 2].imshow(row_vis)
                axes[1, 2].set_title(f"6. Row Detection → {decision} (conf: {confidence})")
                axes[1, 2].axis('off')

                plt.tight_layout()
                plt.show()

                # Additional detailed debug information
                print("=" * 50)
                print("DETAILED DEBUG INFORMATION:")
                print(f"Total keys detected: {len(key_centroids)}")
                print(f"Rows identified: {len(rows)}")

                if len(rows) >= 2 and target_row_idx < len(rows):
                    target_row = rows[target_row_idx]
                    if len(target_row) >= 5:
                        x_coords = target_row[:5, 0]
                        spacings = np.diff(x_coords)
                        normalized = spacings / np.mean(spacings) if len(spacings) > 0 else []

                        print(f"Target row keys: {len(target_row)}")
                        print(f"First 5 key X positions: {x_coords}")
                        print(f"Spacings: {spacings}")
                        print(f"Normalized spacings: {normalized}")
                        print(f"First spacing ratio: {normalized[0] if len(normalized) > 0 else 'N/A'}")
                        print(f"Spacing std: {np.std(normalized) if len(normalized) > 1 else 'N/A'}")
                        print(f"Decision: {decision}")
                        print(f"Confidence score: {confidence}")
                print("=" * 50)

            results.append(f"{decision} (confidence: {confidence})")

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            results.append(f"Error: {str(e)}")

    return results


def analyze_spacing_pattern_signal(spacings):
    """
    Advanced spacing pattern analysis using signal processing techniques
    """
    if len(spacings) < 4:
        return "Unknown", 0

    normalized = spacings / np.mean(spacings)

    # Apply various signal analysis techniques

    # 1. Statistical analysis
    mean_spacing = np.mean(normalized)
    std_spacing = np.std(normalized)
    cv_spacing = std_spacing / mean_spacing  # Coefficient of variation

    # 2. Fourier analysis for pattern regularity
    fft_result = np.fft.fft(normalized)
    fft_magnitude = np.abs(fft_result)

    # Find dominant frequency (excluding DC component)
    if len(fft_magnitude) > 1:
        dominant_freq = np.argmax(fft_magnitude[1:]) + 1
        dominant_strength = fft_magnitude[dominant_freq]
    else:
        dominant_freq = 0
        dominant_strength = 0

    # 3. Autocorrelation for pattern repetition
    autocorr = np.correlate(normalized, normalized, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    # Decision logic based on signal characteristics
    azerty_score = 0
    qwerty_score = 0

    # AZERTY characteristics: less uniform spacing, specific first gap
    if len(normalized) > 0 and normalized[0] < 0.85:
        azerty_score += 2

    if cv_spacing > 0.12:  # Higher variation
        azerty_score += 1

    # QWERTY characteristics: more uniform spacing
    if cv_spacing < 0.1:  # Lower variation
        qwerty_score += 2

    if dominant_strength > np.mean(fft_magnitude) * 1.5:  # Strong pattern
        qwerty_score += 1

    if azerty_score > qwerty_score:
        return "AZERTY", azerty_score
    elif qwerty_score > azerty_score:
        return "QWERTY", qwerty_score
    else:
        return "Unknown", max(azerty_score, qwerty_score)