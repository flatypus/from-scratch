from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter

FILE = "minecraft.jpg"
img = Image.open(FILE)
draw = ImageDraw.Draw(img)
img_arr = np.array(img.convert("L"), dtype=np.float64)
kernel = 5  # kernel size
local_neighbourhood = 9  # must be most prominent corner in this neighborhood
r = 3  # draw circle radius


def shi_tomasi(img_arr):
    # we want to directly compare I(x+1, y) - I(x-1, y) for example, so we do the 2: and :-2 slice
    # then since there are values on either end that don't match, we slice by 1:-1 (and remember to +1 later)
    I_x = (img_arr[1:-1, 2:] - img_arr[1:-1, :-2]) / 2
    I_y = (img_arr[2:, 1:-1] - img_arr[:-2, 1:-1]) / 2

    a = uniform_filter(I_x ** 2, size=kernel) * kernel * kernel
    b = uniform_filter(I_x * I_y, size=kernel) * kernel * kernel
    c = uniform_filter(I_y ** 2, size=kernel) * kernel * kernel

    # rewrite as quadratic and simultaneously solve for smaller of the two eigenvalues (take the -np.sqrt() branch of the + or -)
    l = ((a + c) - np.sqrt((a - c) ** 2 + 4 * (b ** 2))) / 2
    threshold = l.max() * 0.01  # some arbitrary quality level
    local_max = (l == maximum_filter(l, size=local_neighbourhood))
    for y, x in np.argwhere(local_max & (l > threshold)):
        draw.ellipse([x + 1 - r, y + 1 - r, x + 1 + r, y + 1 + r], fill="red")
    img.save(f"dots_{FILE}.png")


shi_tomasi(img_arr)
