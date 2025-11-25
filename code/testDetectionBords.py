import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

# Chemin Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Théo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


# -------------------------------------------------------------
# 1. Détection des bords (Sobel)
# -------------------------------------------------------------
def sobel(gray):
    Sx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    Sy = np.array([
        [1,  2,  1],
        [0,  0,  0],
        [-1, -2, -1]
    ])

    H, W = gray.shape
    G = np.zeros((H, W))

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            region = gray[y-1:y+2, x-1:x+2]
            gx = np.sum(region * Sx)
            gy = np.sum(region * Sy)
            G[y, x] = np.sqrt(gx*gx + gy*gy)

    G = (G / G.max()) * 255
    return G.astype(np.uint8)


# -------------------------------------------------------------
# 2. Détection automatique des bandes horizontales (touches)
# -------------------------------------------------------------
def detect_top_rows(edges):
    mask = edges > 80

    row_strength = np.sum(mask, axis=1)
    row_strength = row_strength / np.max(row_strength)

    rows = np.where(row_strength > 0.25)[0]

    bandes = []
    cur = [rows[0]]
    for i in range(1, len(rows)):
        if rows[i] == rows[i-1] + 1:
            cur.append(rows[i])
        else:
            bandes.append(cur)
            cur = [rows[i]]
    bandes.append(cur)

    return bandes


# -------------------------------------------------------------
# 3. Extraire la bande rangée des lettres (2ème bande)
# -------------------------------------------------------------
def extract_letter_row(img, band):
    y1 = band[0] - 5
    y2 = band[-1] + 50

    y1 = max(0, y1)
    y2 = min(img.shape[0], y2)

    return img[y1:y2]


# -------------------------------------------------------------
# 4. Détecter colonnes de touches dans la bande
# -------------------------------------------------------------
def detect_key_columns_in_band(band_img):
    gray = np.mean(band_img, axis=2)
    edges = sobel(gray)

    mask = edges > 80
    col_strength = np.sum(mask, axis=0)

    threshold = np.percentile(col_strength, 70)
    key_cols = col_strength > threshold

    return key_cols, edges


# -------------------------------------------------------------
# 5. Grouper colonnes -> touches
# -------------------------------------------------------------
def group_columns(mask):
    groups = []
    current = []

    for i, val in enumerate(mask):
        if val:
            current.append(i)
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)

    return groups


# -------------------------------------------------------------
# 6. Extraire première touche
# -------------------------------------------------------------
def extract_first_key(band_img, groups):
    g = groups[0]
    x1, x2 = g[0], g[-1]
    return band_img[:, x1:x2]


# -------------------------------------------------------------
# PROGRAMME DE DEBUG COMPLET
# -------------------------------------------------------------

img = mpimg.imread("data/azerty/AZERTY1.jpg")
gray = np.mean(img, axis=2)
edges = sobel(gray)

# 1. Détection des bandes horizontales
bandes = detect_top_rows(edges)

# 2. On prend la 2e (rangée AZERTY)
letter_band = bandes[1]

band_img = extract_letter_row(img, letter_band)

plt.figure(figsize=(12,3))
plt.imshow(band_img)
plt.title("Rangée des lettres détectée")
plt.show()

# 3. Segmentation horizontale
key_cols, band_edges = detect_key_columns_in_band(band_img)
groups = group_columns(key_cols)

plt.figure(figsize=(12,3))
plt.imshow(band_edges, cmap="gray")
plt.title("Bords dans la rangée des lettres")
plt.show()

plt.figure(figsize=(12,2))
plt.plot(key_cols)
plt.title("Colonnes marquées comme touches")
plt.show()

# 4. Extraire la première touche
first_key = extract_first_key(band_img, groups)

plt.figure(figsize=(3,3))
plt.imshow(first_key)
plt.title("Première touche extraite")
plt.show()

print("Debug terminé.")
