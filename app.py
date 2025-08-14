from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from skimage.morphology import skeletonize, thin, convex_hull_image
from skimage.measure import label, regionprops

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def save_image(name, img):
    path = os.path.join(OUTPUT_FOLDER, name)
    cv2.imwrite(path, img * 255 if img.max() <= 1 else img)
    return path

def save_output(name, img):
    path = os.path.join(OUTPUT_FOLDER, name)
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(path, img)
    return path

def prune_skeleton(skel, size=5):
    pruned = skel.copy()
    labeled = label(skel)
    for region in regionprops(labeled):
        if region.area < size:
            for coord in region.coords:
                pruned[coord[0], coord[1]] = 0
    return pruned

def extract_connected_components(binary_img):
    labeled_img = label(binary_img) # skimage labeling
    color_output = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    np.random.seed(42)
    for region in regionprops(labeled_img):
        color = np.random.randint(0, 255, 3)
        for coord in region.coords:
            color_output[coord[0], coord[1]] = color
    return color_output

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)

            eroded = cv2.erode(binary, kernel, iterations=1)
            boundary = binary - eroded
            
            # Hole filling
            binary_uint8 = (binary * 255).astype(np.uint8)
            inverted = cv2.bitwise_not(binary_uint8.copy())

            # Create mask for floodFill (size needs to be 2 pixels larger)
            h, w = inverted.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)

            # Flood fill from (0, 0)
            cv2.floodFill(inverted, mask, (0, 0), 255)

            # Invert floodfilled image to get the filled holes
            filled_inv = cv2.bitwise_not(inverted)

            # Combine with original image
            hole_filled = cv2.bitwise_or(binary_uint8, filled_inv)

            convex = convex_hull_image(binary)
            thin_img = thin(binary)
            thick_img = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1)
            skeleton = skeletonize(binary)
            pruned = prune_skeleton(skeleton)

            components = extract_connected_components(binary)
            components_path = os.path.join(OUTPUT_FOLDER, 'connected_components.png')
            cv2.imwrite(components_path, components)

            result_paths = {
                'Original': save_image('original.png', binary),
                'Boundary': save_image('boundary.png', boundary),
                #'Hole Filled': save_image('hole_filled.png', hole_filled),
                'Convex Hull': save_image('convex.png', convex),
                'Thinned': save_image('thinned.png', thin_img),
                'Thickened': save_image('thickened.png', thick_img),
                'Skeleton': save_image('skeleton.png', skeleton),
                'Pruned Skeleton': save_image('pruned.png', pruned),
                'Connected Components': components_path
            }

            return render_template('index.html', images=result_paths)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


