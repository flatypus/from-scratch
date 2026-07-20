import cv2
import numpy as np
import json

files = ["ver_001", "ver_002", "ver_003"]
mask = cv2.imread(f"mask.png", cv2.IMREAD_GRAYSCALE)

for filename in files:
    img = cv2.imread(f"{filename}.png")

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    corners = cv2.goodFeaturesToTrack(
        gray_img,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=20,
        mask=mask
    )
    corners = np.int32(corners)
    print(f"Found {len(corners)} corners")

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    cv2.imwrite(f"dots_{filename}.png", img)

    with open(f"dots_{filename}.json", "w") as f:
        json.dump(corners.tolist(), f)
