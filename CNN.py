import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt

# Chargement des données
Datadir = "E:\\pcd\\NEU database\\NEU database"
target_size = (200, 200)

# Mapping categories to numerical labels
category_to_label = {'Cr': 0, 'In': 1, 'Pa': 2, 'PS': 3, 'RS': 4, 'Sc': 5}

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
X_train, X_temp, y_train, y_temp = train_test_split(augmented_data, augmented_labels, test_size=0.3, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# Construction du modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 2)),  # Change input_shape to (200, 200, 2)
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),  # Increase number of filters
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),  # Add another convolutional layer
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),  # Increase number of neurons
    layers.Dropout(0.5),  # Add dropout layer for regularization
    layers.Dense(6, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle avec early stopping
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1,
                    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])


model.save("CNN.keras")

# Évaluation du modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Précision sur l\'ensemble de test:', test_acc)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()






