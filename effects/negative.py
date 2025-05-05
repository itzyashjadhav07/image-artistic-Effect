import cv2
import numpy as np
from PIL import Image

def apply_negative_effect(image_input):
    """
    Applies a negative effect to the input image.

    Args:
        image_input (PIL.Image.Image or np.ndarray): Input image.

    Returns:
        PIL.Image.Image: Image with the negative effect applied.
    """
    # Convert PIL Image to OpenCV format if needed
    if isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        image = image_input
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must have 3 channels (RGB/BGR).")
    else:
        raise TypeError("Input must be a PIL.Image or numpy.ndarray")

    # Ensure the image is valid
    if image is None or image.size == 0:
        raise ValueError("Error: Could not process the image (image is None or empty).")

    # Create the negative image
    negative = 255 - image

    # Convert back to PIL.Image in RGB format
    negative_rgb = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)
    return Image.fromarray(negative_rgb)
