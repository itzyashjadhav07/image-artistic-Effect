from scipy import stats
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict  # Import defaultdict from collections

# Function to update C (centroids)
def update_C(C, histogram):
    while True:
        groups = defaultdict(list)  # Now defaultdict is correctly imported
        for i in range(len(histogram)):
            if histogram[i] == 0:
                continue
            d = np.abs(C - i)
            index = np.argmin(d)
            groups[index].append(i)

        new_C = np.array(C)
        for i, indice in groups.items():
            if np.sum(histogram[indice]) == 0:
                continue
            new_C[i] = int(np.sum(indice * histogram[indice]) / np.sum(histogram[indice]))

        if np.sum(new_C - C) == 0:
            break
        C = new_C
    return C, groups

# K-means histogram clustering function
def KHist(hist):
    alpha = 0.001  # p-value threshold
    N = 80         # minimum group size
    C = np.array([128])

    while True:
        C, groups = update_C(C, hist)
        new_C = set()

        for i, indice in groups.items():
            if len(indice) < N:
                new_C.add(C[i])
                continue

            if len(hist[indice]) >= 8:
                _, pval = stats.normaltest(hist[indice])
            else:
                pval = 1.0

            if pval < alpha:
                left = 0 if i == 0 else C[i - 1]
                right = len(hist) - 1 if i == len(C) - 1 else C[i + 1]
                delta = right - left
                if delta >= 3:
                    new_C.add((C[i] + left) / 2)
                    new_C.add((C[i] + right) / 2)
                else:
                    new_C.add(C[i])
            else:
                new_C.add(C[i])

        if len(new_C) == len(C):
            break
        else:
            C = np.array(sorted(new_C))
    return C


# Function to apply anime effect
def animefy(input_image, old=0):
    output = np.array(input_image)
    x, y, channel = output.shape

    # Apply bilateral filter on each channel
    for i in range(channel):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 50, 50)

    # Edge detection
    edge = cv2.Canny(output, 100, 200)

    # Convert image to HSV
    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    # Create HSV histograms
    hists = [
        np.histogram(output[:, :, 0], bins=181, range=(0, 180))[0],
        np.histogram(output[:, :, 1], bins=256, range=(0, 255))[0],
        np.histogram(output[:, :, 2], bins=256, range=(0, 255))[0]
    ]

    Collect = []
    for h in tqdm(hists, desc="Progress 1 of 2"):
        Collect.append(KHist(h))

    output = output.reshape((-1, channel))
    for i in tqdm(range(channel), desc="Progress 2 of 2"):
        channel1 = output[:, i]
        index = np.argmin(np.abs(channel1[:, np.newaxis] - Collect[i]), axis=1)
        output[:, i] = Collect[i][index]
    output = output.reshape((x, y, channel))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    # Find and draw contours on the filtered image
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, 0, thickness=1)

    return output


# Example usage (to be executed in your main application)
if __name__ == '__main__':
    import argparse
    import time
    import cv2
    import os

    # Argument parser for input and output image paths
    ap = argparse.ArgumentParser(description="Apply anime effect to an image.")
    ap.add_argument("-i", "--image", required=True, help="Path to input image")
    ap.add_argument("-o", "--output", default="assets/anime_effect.jpg", help="Path to save the output image")
    args = vars(ap.parse_args())

    # Load the image
    img = cv2.imread(args["image"])

    # Validate image load
    if img is None:
        print(f"Error: Unable to load the image from '{args['image']}'. Check the path and format.")
        exit()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args["output"]), exist_ok=True)

    # Apply anime effect
    start_time = time.time()
    print("Applying anime effect...")
    output, _, _, _ = animefy(img, old=1)  # Apply the effect
    end_time = time.time()

    # Save the result
    cv2.imwrite(args["output"], output)
    print(f"Done! Your result has been saved to: {args['output']}")
    print(f"Processing time: {end_time - start_time:.2f}s")
