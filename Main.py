import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

Datadir = "C:\Users\wajdi\Downloads\NEU surface defect database"
for img in os.listdir(Datadir):
    img_array = cv2.imread(os.path.join(Datadir,img))
    plt.imshow(img_array)
    plt.show()
    print(img_array)
    break





Datadir = "C:\Users\wajdi\Downloads\NEU surface defect database"
target_size = (200, 200) 
augmented_data = []



Datadir = "C:/Users/pc/pyproj/PCD_1/NEU-CLS"
target_size = (200, 200) 
augmented_data = []

def adjust_brightness_contrast(image, alpha=1.5, beta=50):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)



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


    adjusted_img = adjust_brightness_contrast(img, alpha=1.5, beta=50)

    # Flip the image horizontally
    flipped_img_horizontal = cv2.flip(img, 1)
    
    # Flip the image vertically
    flipped_img_vertical = cv2.flip(img, 0)
    

    # Append original and augmented images to the list
    augmented_data.extend([img, flipped_img_horizontal, flipped_img_vertical])

    # Brightness and Contrast Adjustment
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 30    # Brightness control (0-100)
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Append original and augmented images to the list
    augmented_data.extend([img, adjusted_img, flipped_img_horizontal, flipped_img_vertical])


# Convert the list to a numpy array
augmented_data = np.array(augmented_data)






# Display the first 3 images interactively using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ["Original Image", "Adjusted Image", "Flipped Image"]

for i in range(3):
    axes[i].imshow(cv2.cvtColor(augmented_data[i], cv2.COLOR_BGR2RGB))
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.show()


# Display some of the augmented images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(cv2.cvtColor(augmented_data[i], cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(f'Augmented Image {i+1}')
plt.show()

num_images = len(augmented_data)
print("Number of images in augmented dataset:", num_images)

training_data = []
def create_training_data():
    for img in os.listdir(Datadir):
        img_array = cv2.imread(os.path.join(Datadir,img))
