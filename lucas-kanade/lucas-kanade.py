from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import windows
import json

FILE_1 = "ver_001"
FILE_2 = "ver_002"

with open(f"dots_{FILE_1}.json", "r") as f:
    points = np.array(json.load(f))

frame_1 = Image.open(f"{FILE_1}.png")
frame_2 = Image.open(f"{FILE_2}.png")
draw = ImageDraw.Draw(frame_1)
arr_1 = np.array(frame_1.convert("L"), dtype=np.float64)
arr_2 = np.array(frame_2.convert("L"), dtype=np.float64)

kernel = 9  # kernel size
r = kernel // 2
c = 3  # draw circle radius


def lucas_kanade(img_1, img_2):
    # same as shi-tomasi.py
    I_x = (img_1[1:-1, 2:] - img_1[1:-1, :-2]) / 2
    I_y = (img_1[2:, 1:-1] - img_1[:-2, 1:-1]) / 2
    I_t = (img_2[1:-1, 1:-1] - img_1[1:-1, 1:-1])
    I_x = np.pad(I_x, r, mode='constant')
    I_y = np.pad(I_y, r, mode='constant')
    I_t = np.pad(I_t, r, mode='constant')

    kernel_1d = windows.gaussian(kernel, std=kernel / 4).reshape(kernel, 1)
    W = np.outer(kernel_1d, kernel_1d).flatten()  # flattened kernel

    for point in points:
        x, y = point[0]
        cy, cx = y - 1 + r, x - 1 + r  # - 1 + r to consider padding/slice here
        s = (slice(cy - r, cy + r + 1), slice(cx - r, cx + r + 1))
        patch_Ix, patch_Iy, patch_It = I_x[s], I_y[s], I_t[s]
        if patch_Ix.shape != (kernel, kernel):
            continue
        A = np.stack([patch_Ix.flatten(), patch_Iy.flatten()], axis=1)
        b = np.stack([patch_It.flatten()], axis=1)
        A_TWA = (A.T * W) @ A
        A_TWb = -((A.T * W) @ b)
        dx, dy = np.linalg.solve(A_TWA, A_TWb).flatten()
        draw.ellipse([x + 1 - c, y + 1 - c, x + 1 + c, y + 1 + c], fill="red")
        draw.line([(x, y), (x + dx, y + dy)], fill="black", width=1)
    frame_1.save(f"arrow_{FILE_1}.png")


lucas_kanade(arr_1, arr_2)
