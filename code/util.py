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

    gx = (
            gray[:-2, :-2] * Sx[0, 0] + gray[:-2, 1:-1] * Sx[0, 1] + gray[:-2, 2:] * Sx[0, 2] +
            gray[1:-1, :-2] * Sx[1, 0] + gray[1:-1, 1:-1] * Sx[1, 1] + gray[1:-1, 2:] * Sx[1, 2] +
            gray[2:, :-2] * Sx[2, 0] + gray[2:, 1:-1] * Sx[2, 1] + gray[2:, 2:] * Sx[2, 2]
    )

    gy = (
            gray[:-2, :-2] * Sy[0, 0] + gray[:-2, 1:-1] * Sy[0, 1] + gray[:-2, 2:] * Sy[0, 2] +
            gray[1:-1, :-2] * Sy[1, 0] + gray[1:-1, 1:-1] * Sy[1, 1] + gray[1:-1, 2:] * Sy[1, 2] +
            gray[2:, :-2] * Sy[2, 0] + gray[2:, 1:-1] * Sy[2, 1] + gray[2:, 2:] * Sy[2, 2]
    )

    # Normalize to 0-255
    G = np.sqrt(gx * gx + gy * gy)
    G = (G - G.min()) / (G.max() - G.min()) * 255
    return G.astype(np.uint8)