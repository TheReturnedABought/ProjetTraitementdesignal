import cv2
import numpy as np
import matplotlib.pyplot as plt


def upscale_image(img, scale=2, interpolation=cv2.INTER_CUBIC):
    """
    Upscales an image by a given factor safely.

    Parameters:
        img (numpy.ndarray): Input image.
        scale (float): Upscaling factor.
        interpolation (int): Interpolation method.

    Returns:
        numpy.ndarray: Upscaled image.
    """
    height, width = img.shape[:2]

    # Compute new size
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Ensure we don't exceed OpenCV's limit
    SHRT_MAX = 30000
    new_width = min(new_width, SHRT_MAX - 1)
    new_height = min(new_height, SHRT_MAX - 1)

    new_size = (new_width, new_height)

    upscaled = cv2.resize(img, new_size, interpolation=interpolation)
    return upscaled

def increase_contrast(img, clip_limit=1.0, tile_grid_size=(5, 5)):
    """
    Increases the contrast of an image using CLAHE (adaptive histogram equalization).

    Parameters:
        img (numpy.ndarray): Input image (BGR or grayscale).
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of the grid for the histogram equalization.

    Returns:
        numpy.ndarray: Image with enhanced contrast.
    """
    # Convert to grayscale if image is BGR
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)

    # If original was color, convert back to BGR
    if len(img.shape) == 3:
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr
    else:
        return enhanced

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def apply_gaussian_blur(img, kernel_size=(5, 5), sigma=0):
    """
    Applies Gaussian Blur to reduce noise while preserving edges.

    Parameters:
        img (numpy.ndarray): Input image (BGR or grayscale)
        kernel_size (tuple): Size of the Gaussian kernel (must be odd numbers)
        sigma (int): Standard deviation in X and Y direction (0 = calculated from kernel size)

    Returns:
        numpy.ndarray: Blurred image
    """
    return cv2.GaussianBlur(img, kernel_size, sigma)

def convert_to_gray(img):
    """
    Converts a BGR image to Otsu-binarized grayscale.
    If the result is mostly black, it automatically inverts it.
    """
    # Convert BGR → grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary



def display_results(original_img):
    plt.figure(figsize=(18, 6))
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.show()

def draw_boxes_on_image(img, detections):
    """Dessine les bounding boxes sur une copie de l'image."""
    if len(img.shape) == 2:
        img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_display = img.copy()

    for bbox, text, score in detections:
        # Convertir bbox en points entiers
        pts = np.array(bbox, dtype=np.int32)

        # Dessiner le polygone (bounding box)
        cv2.polylines(img_display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Ajouter le texte au-dessus de la box
        cv2.putText(
            img_display,
            f"{text}",
            (int(bbox[0][0]), int(bbox[0][1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return img_display