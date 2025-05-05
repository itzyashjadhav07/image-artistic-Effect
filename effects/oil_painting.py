import cv2
import numpy as np
from PIL import Image

def apply_oil_painting_effect(image_input, size=800):
    """
    Applies an oil painting effect to the input image.

    Args:
        image_input (PIL.Image.Image or np.ndarray): Input image.
        size (int): Resize width (maintains aspect ratio).

    Returns:
        PIL.Image.Image: Image with oil painting effect.
    """
    # Convert PIL Image to OpenCV format if needed
    if isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError("Input must be a PIL.Image or numpy.ndarray")

    if image is None:
        raise ValueError("Invalid image input (None)")

    # Resize image while maintaining aspect ratio
    height = int((size / float(image.shape[1])) * image.shape[0])
    resized_img = cv2.resize(image, (size, height), interpolation=cv2.INTER_AREA)

    # Check if OpenCV 'xphoto' module is available
    if not hasattr(cv2, 'xphoto'):
        raise ImportError("OpenCV's 'xphoto' module is required for the oil painting effect.")

    # Apply the oil painting effect
    oil_painting_img = cv2.xphoto.oilPainting(resized_img, 7, 1)

    # Convert back to PIL Image for output
    oil_painting_rgb = cv2.cvtColor(oil_painting_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(oil_painting_rgb)
