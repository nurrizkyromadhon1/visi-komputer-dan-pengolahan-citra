import numpy as np
import cv2
from sklearn.cluster import KMeans

# Load an image (replace 'your_image.jpg' with the actual image file)
image_path = 'bola2.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image.reshape((-1, 3))

# Apply ISODATA clustering (K-Means with dynamic cluster adaptation)
def isodata_clustering(data, max_clusters, min_samples=5, max_iter=100):
    kmeans = KMeans(n_clusters=max_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    for _ in range(max_iter):
        cluster_sizes = np.bincount(labels)
        mask = cluster_sizes > min_samples
        if mask.sum() == 0:
            break

        kmeans = KMeans(n_clusters=mask.sum(), random_state=42, init=centers[mask])
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

    return labels, centers

# Apply ISODATA clustering
num_clusters = 6  # Adjust the number of clusters as needed
isodata_labels, isodata_centers = isodata_clustering(pixels, num_clusters)

# Replace each pixel with its corresponding cluster center
segmented_image = isodata_centers[isodata_labels].reshape(image.shape)

# Display the original and segmented images
cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imshow('Segmented Image', cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
