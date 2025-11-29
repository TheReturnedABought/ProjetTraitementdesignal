import numpy as np

def sobel(gray):
    Sx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    Sy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    H, W = gray.shape
    G = np.zeros((H, W))

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            region = gray[y - 1:y + 2, x - 1:x + 2]
            gx = np.sum(region * Sx)
            gy = np.sum(region * Sy)
            G[y, x] = np.sqrt(gx * gx + gy * gy)

    # Normalize to 0-255
    G = (G / G.max()) * 255
    return G.astype(np.uint8)