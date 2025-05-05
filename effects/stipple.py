# stipple.py

from PIL import Image
from tqdm import tqdm
import random

def stippler(img, width, height):
    imgNew = Image.new('L', (width, height))
    for x in tqdm(range(width)):
        for y in range(height):
            gray_value = img.getpixel((x, y))
            randNum = random.randint(0, 255)
            if randNum >= gray_value:
                imgNew.putpixel((x, y), 0)
            else:
                imgNew.putpixel((x, y), 255)
    return imgNew

# ✅ Protect CLI logic from being run during import
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to input image")
    ap.add_argument("-o", "--output", default="assets/Stippled.png", help="Path to save output image")
    args = ap.parse_args()

    # Open and convert image
    img = Image.open(args.image).convert('L')
    width, height = img.size

    # Apply stippling effect
    stip_img = stippler(img, width, height)

    # Save result
    stip_img.save(args.output)
