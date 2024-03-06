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
    
    # Read the image with error handling
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        continue
    
    # Resize the image to the target size
    img = cv2.resize(img, target_size)
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    
    # Flip the thresholded image horizontally
    flipped_img_horizontal = cv2.flip(thresh_img, 1)
    
    # Flip the thresholded image vertically
    flipped_img_vertical = cv2.flip(thresh_img, 0)
    
    # Append original and augmented images to the list
    augmented_data.extend([img, flipped_img_horizontal, flipped_img_vertical])

# Convert the list to a numpy array
augmented_data = np.array(augmented_data)

# Display some of the augmented images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Show image number 97
axes[0].imshow(augmented_data[1], cmap='gray')
axes[0].axis('off')
axes[0].set_title('Augmented Image 97')

# Show image number 98
axes[1].imshow(augmented_data[2], cmap='gray')
axes[1].axis('off')
axes[1].set_title('Augmented Image 98')

plt.show()



num_images = len(augmented_data)
print("Number of images in augmented dataset:", num_images)



import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

Datadir = "C:/Users/pc/pyproj/PCD_1/NEU-CLS"
target_size = (200, 200)
original_images = []
thresholded_images = []

# Name of the image to display
image_name = "SteelDefect (294)"

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
        
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh_img = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
        
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
axes[0].set_title('Original Image')

# Show thresholded image
axes[1].imshow(thresholded_images[0], cmap='gray')
axes[1].axis('off')
axes[1].set_title('Thresholded Image')

plt.show()

