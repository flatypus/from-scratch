from PIL import Image, ImageDraw
import numpy as np

BOUNDS = 500
POINTS = 200
SCALE = 200
ERR_SCALE = 20

img = Image.new(mode="RGB", size=(BOUNDS, BOUNDS))
draw = ImageDraw.Draw(img)


def dot(x, y, r=3):
    draw.ellipse([x - r, y - r, x + r, y + r], fill="red")


# set up
slope = np.random.uniform()
rng = np.random.default_rng()
points = []

for ex, ey, scalar in rng.normal(loc=0, scale=0.5, size=(POINTS, 3)):
    dx, dy = scalar, slope * scalar
    cx, cy = BOUNDS / 2 + dx * SCALE, BOUNDS / 2 + dy * SCALE
    x, y = cx + ex * ERR_SCALE, cy + ey * ERR_SCALE
    points.append((x, y))
    dot(x, y)


# RANSAC implementation
points = np.array(points)
best_fit = None
best_case = -1

ITERS = 10
DEVIATION = 10
for _ in range(ITERS):
    c1, c2 = rng.choice(points, size=2, replace=False)
    c1x, c1y = c1
    c2x, c2y = c2
    include = 0
    ax, ay = points[:, 0] - c1x, points[:, 1] - c1y
    bx, by = c2x - c1x, c2y - c1y
    distances = np.abs(ax * by - ay * bx) / np.linalg.norm((bx, by))
    include = np.count_nonzero(distances < DEVIATION)
    if include > best_case:
        best_case = include
        best_fit = by/bx

print(f"Real slope: {slope}")
print(f"Slope: {best_fit}")

s = BOUNDS
dotmin, dotmax = BOUNDS // 2 - s, BOUNDS // 2 + s
draw.line([(dotmin, dotmin * slope), (dotmax, dotmax * slope)], fill="green", width=10)
draw.line([(dotmin, dotmin * best_fit), (dotmax, dotmax * best_fit)], fill="blue", width=5)
img.save("dots.png")
