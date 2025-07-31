# Shape-Analysis-and-Morphological-Processing-using-Computer-Vision
This project implements a computer vision pipeline to analyze shapes and perform morphological operations on binary images. Users can upload grayscale or binary images to extract meaningful features like boundaries, convex hulls, skeletons, and connected components. The tool is built using Flask and OpenCV, with additional shape analysis powered by skimage.
Morphological operations such as erosion, dilation, thinning, thickening, skeletonization, pruning, hole filling, and connected component labeling are demonstrated visually. The app is especially useful for educational purposes and for preliminary analysis in applications like medical imaging, document processing, and industrial inspection.

📂 Features
Upload and process binary or grayscale images
Boundary extraction via erosion
Convex Hull extraction using skimage
Thinning and thickening operations
Skeletonization and pruning of small branches
Connected components labeling with random color mapping
Real-time visualization of processed outputs

📌 Use Cases
Educational visualization of morphological techniques
Pre-processing step in object detection or segmentation
Medical image analysis (e.g., blood cell shapes)
Document analysis (e.g., handwriting, text blobs)


🧪 Examples of Morphological Operations

| Operation            | Description                            |
| -------------------- | -------------------------------------- |
| Erosion              | Removes pixels on object boundaries    |
| Dilation             | Adds pixels to object boundaries       |
| Boundary             | Original minus eroded                  |
| Convex Hull          | Smallest convex shape enclosing object |
| Thinning             | Reduces width to single-pixel lines    |
| Skeletonization      | Core structure of the object           |
| Pruning              | Removes short branches from skeleton   |
| Connected Components | Labels and visualizes distinct regions |
