import cv2
import numpy as np
from PIL import Image
import random

def apply_low_poly_effect(image_input):
    """Apply low poly effect to the image."""
    
    # Check if the input is a NumPy array (image loaded into memory)
    if isinstance(image_input, np.ndarray):
        img = image_input  # Directly use the NumPy array
    else:
        # If it's a file path, open it using PIL
        img = Image.open(image_input)
        img = np.array(img)  # Convert it to a NumPy array

    # Convert to RGB if the image is in BGR format (as OpenCV uses BGR)
    if img.shape[-1] == 3:  # Check if it's a color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to a smaller size for faster processing (optional)
    target_width = 480
    height, width = img.shape[:2]
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    img = cv2.resize(img, (target_width, new_height))

    # Convert the image to a set of polygons
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Perform edge detection using Canny
    edges = cv2.Canny(img_gray, 100, 200)

    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank image to draw the low poly effect on
    low_poly_img = np.zeros_like(img)

    # Randomly select colors to fill polygons
    for contour in contours:
        if len(contour) > 5:  # Avoid drawing very small contours
            # Find the convex hull of the contour to get the polygon
            polygon = cv2.convexHull(contour)

            # Random color for each polygon
            color = [random.randint(0, 255) for _ in range(3)]
            
            # Fill the polygon with the random color
            cv2.fillPoly(low_poly_img, [polygon], color)

    # Return the processed image
    return low_poly_img
