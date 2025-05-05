from PIL import Image, ImageDraw, UnidentifiedImageError
from tqdm import tqdm
import numpy as np

def get_pixel(image, x, y):
    width, height = image.size
    if x >= width or y >= height or x < 0 or y < 0:
        return None
    return image.getpixel((x, y))

def color_average(image, x0, y0, x1, y1):
    red, green, blue = 0, 0, 0
    count = 0

    for x in range(x0, x1):
        for y in range(y0, y1):
            pixel = get_pixel(image, x, y)
            if pixel:
                r, g, b = pixel[:3]
                red += r
                green += g
                blue += b
                count += 1

    if count == 0:
        return (255, 255, 255)

    return (red // count, green // count, blue // count)

def convert_pointillize(image, radius=6):
    width, height = image.size
    output = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(output)

    jitter = [1, 0, 1, 1, 2, 3, 3, 1, 2, 1]
    count = 0

    for x in tqdm(range(0, width, radius + 3), desc="Applying Pointillism"):
        for y in range(0, height, radius + 3):
            color = color_average(image, x - radius, y - radius, x + radius, y + radius)
            jitter_x = jitter[count % len(jitter)]
            count += 1
            jitter_y = jitter[count % len(jitter)]
            count += 1
            draw.ellipse((x - radius + jitter_x, y - radius + jitter_y, x + radius + jitter_x, y + radius + jitter_y),
                         fill=color)

    return output

def apply_pointillism_effect(image_input, size=800):
    """
    Applies a pointillism effect to a PIL image.

    Args:
        image_input (PIL.Image.Image): Input image.
        size (int): Desired width, aspect ratio preserved.

    Returns:
        PIL.Image.Image: Pointillized image.
    """
    if not isinstance(image_input, Image.Image):
        raise TypeError("Input must be a PIL.Image.Image")

    width, height = image_input.size
    new_height = int((size / width) * height)
    resized = image_input.resize((size, new_height))

    return convert_pointillize(resized)
