# effects/comic_effect.py

import cv2
import numpy as np
from tqdm import tqdm
import os

def comic(img):
    with tqdm(total=100, desc="Applying Comic Effect") as pbar:
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pbar.update(10)

        edgesOnly = cv2.Canny(grayImg, 50, 150)
        pbar.update(20)

        color = cv2.bilateralFilter(img, 9, 300, 300)
        pbar.update(30)

        edgesOnlyInv = cv2.bitwise_not(edgesOnly)
        edgesOnlyInv = cv2.cvtColor(edgesOnlyInv, cv2.COLOR_GRAY2BGR)

        cartoon = cv2.addWeighted(color, 0.9, edgesOnlyInv, 0.3, 0)
        pbar.update(20)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        cartoon = cv2.filter2D(cartoon, -1, kernel)
        pbar.update(20)

        pbar.update(10)

    return cartoon

# ✅ Safe CLI block for testing only
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Apply comic cartoon effect to an image.")
    ap.add_argument("-i", "--image", required=True, help="Path to input image")
    ap.add_argument("-o", "--output", default="assets/comic_cartoon_effect.jpg", help="Path to save the output image")
    args = vars(ap.parse_args())

    img = cv2.imread(args["image"])

    if img is None:
        print(f"Error: Unable to load the image from '{args['image']}'. Check the path and format.")
        exit()

    os.makedirs(os.path.dirname(args["output"]), exist_ok=True)

    res_img = comic(img)

    cv2.imwrite(args["output"], res_img)
    print(f"Done! Your result has been saved to: {args['output']}")
