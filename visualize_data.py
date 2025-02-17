import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astroNN.datasets import galaxy10

# Load dataset
images, labels = galaxy10.load_data()

# Show sample images
def show_sample_images(images, labels, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off")

    plt.show()

show_sample_images(images, labels)

# Show class distribution
unique, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(8, 4))
sns.barplot(x=unique, y=counts)
plt.xlabel("Class Labels")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.show()
