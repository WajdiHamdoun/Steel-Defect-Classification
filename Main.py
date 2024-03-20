import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Chargement des données
Datadir = "E:\\pcd\\NEU database\\NEU database"

target_size = (200, 200) 

# Mapping categories to numerical labels
category_to_label = {'Cr': 0, 'In': 1, 'Pa': 2, 'PS': 3, 'RS': 4, 'Sc': 5}

# Placeholder for labels
labels = []

# Placeholder for augmented data
augmented_data = []

# Boucle à travers chaque image dans le répertoire du dataset
for img_name in os.listdir(Datadir):
    img_path = os.path.join(Datadir, img_name)
    
    # Lecture de l'image avec gestion des erreurs
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        continue
    
    
    # Redimensionnement de l'image à la taille cible
    img = cv2.resize(img, target_size)
    
    # Augmentation de contraste de l'image
    alpha = 1.0  # Contrôle du contraste (1.0-3.0)
    beta = 0  # Contrôle de la luminosité (0-100)
    enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    # Conversion de l'image en niveaux de gris
    gray_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        
    # Application de la binarisation adaptative
    thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 5)
        
    # Extraire la catégorie à partir du nom de l'image
    category = img_name.split()[0]  # Supposant que la catégorie est la première partie du nom de l'image
    
    # Assigner une étiquette numérique à la catégorie
    label = category_to_label[category]
    
    # Ajouter l'étiquette à la liste des étiquettes
    labels.append(label)
    
    # Ajouter les données augmentées à la liste
    augmented_data.extend([gray_img, thresh_img])

# Conversion des listes en tableaux numpy
augmented_data = np.array(augmented_data)
labels = np.array(labels)

# Division des données en ensembles d'entraînement (70%), de validation (15%) et de test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(augmented_data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Construction du modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')  # 6 classes, utilisation de softmax pour la classification multi-classe
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Évaluation du modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Précision sur l\'ensemble de test:', test_acc)



































#training
import os
import cv2
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# Chargement des données
Datadir = "C:/Users/pc/Desktop/NEU surface defect database"
target_size = (200, 200) 

# Mapping categories to numerical labels
category_to_label = {'Cr': 0, 'In': 1, 'Pa': 2, 'PS': 3, 'RS': 4, 'Sc': 5}

# Placeholder for labels
labels = []

# Placeholder for augmented data
# Placeholder for augmented data
augmented_data = []
augmented_labels = []

# Boucle à travers chaque image dans le répertoire du dataset
for img_name in os.listdir(Datadir):
    img_path = os.path.join(Datadir, img_name)
    
    # Lecture de l'image avec gestion des erreurs
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        continue
    
    # Redimensionnement de l'image à la taille cible
    img = cv2.resize(img, target_size)
    
    # Augmentation de contraste de l'image
    alpha = 1.0  # Contrôle du contraste (1.0-3.0)
    beta = 0  # Contrôle de la luminosité (0-100)
    enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    # Conversion de l'image en niveaux de gris
    gray_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        
    # Application de la binarisation adaptative
    thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 5)
        
    # Extraire la catégorie à partir du nom de l'image
    category = img_name.split('_')[0]  # Extracting category from the image name
    if category not in category_to_label:
        print(f"Category '{category}' not found in category_to_label dictionary.")
        continue
    
    # Assigner une étiquette numérique à la catégorie
    label = category_to_label[category]
    
    # Stack the augmented images together along the channel axis
    stacked_img = np.stack([gray_img, thresh_img], axis=-1)
    
    # Ajouter les données augmentées et les étiquettes correspondantes
    augmented_data.append(stacked_img)
    augmented_labels.append(label)
    
# Convert lists to numpy arrays
augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)



# Division des données en ensembles d'entraînement (70%), de validation (15%) et de test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(augmented_data, augmented_labels, test_size=0.3, random_state=50)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=50)

# Construction du modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 2)),  # Change input_shape to (200, 200, 2)
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')  # 6 classes, utilisation de softmax pour la classification multi-classe
])


# Compilation du modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=18, batch_size=64, validation_data=(X_val, y_val))

# Évaluation du modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Précision sur l\'ensemble de test:', test_acc)


#Affichage

# Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


