from PIL import Image, ImageDraw
import numpy as np


FILE = "minecraft.jpg"
img = Image.open(FILE).convert("L")
draw = ImageDraw.Draw(img)
img_arr = np.array(img)
height, width = img_arr.shape
k = 7  # kernel size
radius = 1
epsilon = 1e-2


def get(x, y):
    if not (0 <= x < width) or not (0 <= y < height):
        return 0
    return img_arr[y][x]


def getPatch(start_x, start_y):
    start, end = -(k//2), k//2
    a, b, c = 0, 0, 0
    for dy in range(start, end + 1):
        for dx in range(start, end + 1):
            x, y = start_x + dx, start_y + dy
            i_x = (get(x + 1, y) - get(x - 1, y)) / 2
            i_y = (get(x, y + 1) - get(x, y - 1)) / 2
            a += i_x ** 2
            b += i_x * i_y
            c += i_y ** 2
    return a, b, c


for y in range(height):
    print(y / height)
    for x in range(width):
        a, b, c = getPatch(x, y)
        l = (a + c) - np.sqrt((a - c) ** 2 + 4 * (b ** 2))
        if l > epsilon:
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, fill="red")

img.save(f"dots_{FILE}")
