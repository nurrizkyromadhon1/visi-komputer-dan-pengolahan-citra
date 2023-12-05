import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import feature
import matplotlib.pyplot as plt

# Load an image (replace 'your_image.jpg' with the actual image file)
image_path = 'bola2.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Extract Local Binary Pattern (LBP) features
radius = 3
n_points = 8 * radius
lbp_features = feature.local_binary_pattern(image, n_points, radius, method="uniform")

# Reshape the LBP features to a 1D array
lbp_features_1d = lbp_features.flatten()

# Reshape the image to a 1D array for clustering
image_1d = image.flatten()

# Concatenate LBP features with pixel values
data = np.column_stack((image_1d, lbp_features_1d))

# Apply K-Means clustering
num_clusters = 3  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(data)

# Reshape the labels to the shape of the original image
clustered_image = labels.reshape(image.shape)

# Display the original and clustered images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')

ax2.imshow(clustered_image, cmap='viridis')
ax2.set_title('Clustered Image')

plt.show()
