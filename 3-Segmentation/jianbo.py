import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation, color, io

# Load an image (replace 'your_image.jpg' with the actual image file)
image_path = 'your_image.jpg'
image = io.imread(image_path)

# Convert the image to a 2D array (graph representation)
graph = color.rgb2gray(image)

# Apply Normalized Cut for image segmentation
labels = segmentation.slic(image, compactness=30, n_segments=400)
segmented_image = color.label2rgb(labels, image, kind='avg')

# Display the original and segmented images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image)
ax1.set_title('Original Image')

ax2.imshow(segmented_image)
ax2.set_title('Segmented Image')

plt.show()
