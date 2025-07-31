import streamlit as st
import cv2
import numpy as np
from skimage.morphology import skeletonize, thin, convex_hull_image
from skimage.measure import label, regionprops
from PIL import Image

st.set_page_config(page_title="Morphological Operations", layout="wide")
st.title("🧠 Morphological Operations on Binary Images")

def prune_skeleton(skel, size=5):
    pruned = skel.copy()
    labeled = label(skel)
    for region in regionprops(labeled):
        if region.area < size:
            for coord in region.coords:
                pruned[coord[0], coord[1]] = 0
    return pruned

def extract_connected_components(binary_img):
    labeled_img = label(binary_img)
    color_output = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    np.random.seed(42)
    for region in regionprops(labeled_img):
        color = np.random.randint(0, 255, 3)
        for coord in region.coords:
            color_output[coord[0], coord[1]] = color
    return color_output

uploaded_file = st.file_uploader("Upload a binary image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)

    # Boundary Extraction
    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary = binary - eroded

    # Hole Filling
    binary_uint8 = (binary * 255).astype(np.uint8)
    inverted = cv2.bitwise_not(binary_uint8.copy())
    h, w = inverted.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inverted, mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(inverted)
    hole_filled = cv2.bitwise_or(binary_uint8, filled_inv)

    # Other Morphological Operations
    convex = convex_hull_image(binary)
    thin_img = thin(binary)
    thick_img = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1)
    skeleton = skeletonize(binary)
    pruned = prune_skeleton(skeleton)
    components = extract_connected_components(binary)

    # Display all outputs
    col1, col2 = st.columns(2)
    col1.image(binary * 255, caption="Original Binary", use_column_width=True)
    col2.image(boundary * 255, caption="Boundary Extraction", use_column_width=True)

    col1, col2 = st.columns(2)
    col1.image(hole_filled, caption="Hole Filled", use_column_width=True)
    col2.image(convex * 255, caption="Convex Hull", use_column_width=True)

    col1, col2 = st.columns(2)
    col1.image(thin_img * 255, caption="Thinned", use_column_width=True)
    col2.image(thick_img * 255, caption="Thickened", use_column_width=True)

    col1, col2 = st.columns(2)
    col1.image(skeleton * 255, caption="Skeleton", use_column_width=True)
    col2.image(pruned * 255, caption="Pruned Skeleton", use_column_width=True)

    st.image(components, caption="Connected Components", use_column_width=True)
