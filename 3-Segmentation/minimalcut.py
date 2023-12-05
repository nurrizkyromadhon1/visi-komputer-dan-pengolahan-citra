import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load an image (replace 'your_image.jpg' with the actual image file)
image_path = 'image1.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a mask (0: sure background, 2: sure foreground)
mask = np.zeros(image.shape[:2], np.uint8)

# Define a rectangle around the object of interest (rect = (start_x, start_y, width, height))
rect = (50, 50, 300, 200)

# Initialize the background and foreground models
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Apply GrabCut algorithm
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to get the segmented image
segmentation_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
segmented_image = image * segmentation_mask[:, :, np.newaxis]

# Display the original and segmented images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image)
ax1.set_title('Original Image')

ax2.imshow(segmented_image)
ax2.set_title('Segmented Image')

plt.show()
