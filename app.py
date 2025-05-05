import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# === Import your effect functions ===
from effects.anime_effect import animefy
from effects.comic_cartoon_effect import comic
from effects.low_poly import apply_low_poly_effect
from effects.negative import apply_negative_effect
from effects.oil_painting import apply_oil_painting_effect
from effects.pencil_sketch import apply_pencil_sketch_effect
from effects.pointilism import apply_pointillism_effect
from effects.sepia_effect import apply_sepia_effect
from effects.stipple import stippler

# === Helper functions ===
def cv2_to_pil(cv2_img):
    """Convert OpenCV BGR image to PIL RGB."""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_img):
    """Convert PIL RGB image to OpenCV BGR."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def pil_gray_to_cv2_rgb(pil_gray_img):
    """Convert PIL grayscale image to OpenCV RGB format."""
    return cv2.cvtColor(np.array(pil_gray_img.convert("RGB")), cv2.COLOR_RGB2BGR)

# === Streamlit UI ===
st.title("🎨 Image Artistic Effects")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image as OpenCV BGR
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize for consistent performance
    target_width = 480
    height, width = img.shape[:2]
    new_height = int(target_width * height / width)
    img = cv2.resize(img, (target_width, new_height))

    # Effect selection
    effect = st.selectbox("Select an Artistic Effect", [
        "Anime Effect",
        "Comic Cartoon Effect",
        "Low Poly Effect",
        "Negative Effect",
        "Oil Painting",
        "Pencil Sketch",
        "Pointilism Effect",
        "Sepia Effect",
        "Stipple Effect"
    ])

    # === Apply Effects ===
    if effect == "Anime Effect":
        processed_img = animefy(img)

    elif effect == "Comic Cartoon Effect":
        processed_img = comic(img)

    elif effect == "Low Poly Effect":
        processed_img = apply_low_poly_effect(img)

    elif effect == "Negative Effect":
        processed_pil = apply_negative_effect(img)
        processed_img = pil_to_cv2(processed_pil)

    elif effect == "Oil Painting":
        processed_pil = apply_oil_painting_effect(img)
        processed_img = pil_to_cv2(processed_pil)

    elif effect == "Pencil Sketch":
        bw_pil, _ = apply_pencil_sketch_effect(img)
        processed_img = pil_to_cv2(bw_pil)

    elif effect == "Pointilism Effect":
        pil_img = cv2_to_pil(img)
        processed_pil = apply_pointillism_effect(pil_img)
        processed_img = pil_to_cv2(processed_pil)

    elif effect == "Sepia Effect":
        pil_img = cv2_to_pil(img)
        processed_pil = apply_sepia_effect(pil_img)
        processed_img = pil_to_cv2(processed_pil)

    elif effect == "Stipple Effect":
        pil_gray = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        processed_pil = stippler(pil_gray, pil_gray.width, pil_gray.height)
        processed_img = pil_to_cv2(processed_pil)

    # === Show Result ===
    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption=effect, use_column_width=True)

    # === Download Button ===
    st.download_button(
        label="Download Image",
        data=cv2.imencode('.jpg', processed_img)[1].tobytes(),
        file_name=f"{effect.replace(' ', '_').lower()}_output.jpg",
        mime="image/jpeg"
    )
