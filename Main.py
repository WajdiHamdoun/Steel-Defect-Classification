# main

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

Datadir = "C:/Users/pc/pyproj/PCD_1/NEU-CLS"
target_size = (200, 200) 
augmented_data = []

# Loop through each image in the dataset directory
for img_name in os.listdir(Datadir):
    img_path = os.path.join(Datadir, img_name)
    
    # Read the image with error handling (Thumbs.db inexistant pour le mm)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        continue
    
    # Resize the image to the target size (to be adaptive for any input)
    img = cv2.resize(img, target_size)
    
    # Increase contrast of the image
    alpha = 1.0  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    # Convert the image to grayscale (to be adaptive for any input)
    gray_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        
    # Apply adaptive thresholding
    thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 5)
        
    # Flip the image horizontally
    # flipped_img_horizontal = cv2.flip(enhanced_img, 1)
    
    # Flip the image vertically
    # flipped_img_vertical = cv2.flip(enhanced_img, 0)
    
    # Append original image, flipped images, and thresholded image to the list
    augmented_data.extend([gray_img, thresh_img])

# Convert the list to a numpy array
augmented_data = np.array(augmented_data)

# Display some of the augmented images
fig, axes = plt.subplots(5, 4, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(cv2.cvtColor(augmented_data[i], cv2.COLOR_BGR2RGB))
    ax.axis('off')
    if i % 2 == 0 :
        ax.set_title(f'Original Image {((i+1)//2)+1}')
    else :
        ax.set_title(f'Thresholded Image {((i+1)//2)}')
plt.show()

num_images = len(augmented_data)
print("Number of images in augmented dataset:", num_images)





























#Test img per img

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

Datadir = "C:/Users/pc/pyproj/PCD_1/NEU-CLS"
target_size = (200, 200)
original_images = []
thresholded_images = []

# Name of the image to display
image_name = "SteelDefect (1673)"

# Loop through each image in the dataset directory
for img_name in os.listdir(Datadir):
    if img_name.startswith(image_name):
        img_path = os.path.join(Datadir, img_name)
        
        # Read the image with error handling
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_path}")
            continue
        
        # Resize the image to the target size
        img = cv2.resize(img, target_size)
        alpha = 1.0  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 5)
        
        # Append original and thresholded images to their respective lists
        original_images.append(img)
        thresholded_images.append(thresh_img)

# Convert the lists to numpy arrays
original_images = np.array(original_images)
thresholded_images = np.array(thresholded_images)

# Display the image before and after thresholding
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Show original image
axes[0].imshow(original_images[0], cmap='gray')
axes[0].axis('off')
axes[0].set_title(image_name)

# Show thresholded image
axes[1].imshow(thresholded_images[0], cmap='gray')
axes[1].axis('off')
axes[1].set_title('Thresholded Image')

plt.show()


