import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    category = img_name.split('_')[0]  # Supposant que la catégorie est la première partie du nom de l'image
    
    # Assigner une étiquette numérique à la catégorie
    label = category_to_label[category]
    
    # Ajouter l'étiquette à la liste des étiquettes
    labels.append(label)
    
    # Ajouter les données augmentées à la liste
    augmented_data.extend([gray_img, thresh_img])

# Conversion des listes en tableaux numpy
augmented_data = np.array(augmented_data)
labels = np.array(labels)

# Regroupement des données et des étiquettes
data_with_labels = list(zip(augmented_data, labels))

# Division des données en ensembles d'entraînement (70%), de validation (15%) et de test (15%)
train_data, test_data = train_test_split(data_with_labels, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Séparation des données d'entraînement, de validation et de test et de leurs étiquettes
X_train, y_train = zip(*train_data)
X_val, y_val = zip(*val_data)
X_test, y_test = zip(*test_data)

# Conversion des listes en tableaux numpy
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Conversion des données en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).long()
X_val_tensor = torch.tensor(X_val).float()
y_val_tensor = torch.tensor(y_val).long()
X_test_tensor = torch.tensor(X_test).float()
y_test_tensor = torch.tensor(y_test).long()

# Construction du modèle CNN avec PyTorch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calcul de la taille de l'entrée de la couche entièrement connectée
        self.fc_input_size = self.calculate_fc_input_size()
        
        self.fc1 = nn.Linear(self.fc_input_size, 64)
        self.fc2 = nn.Linear(64, 6)  # 6 classes, utilisant softmax pour la classification multi-classe

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        
        # Ajustement de la taille de l'entrée pour la couche entièrement connectée
        x = x.view(-1, self.fc_input_size)
        
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def calculate_fc_input_size(self):
        # Calcul de la taille de l'entrée à partir de la taille de l'image d'entrée
        # et des opérations de pooling et de convolution
        x = torch.randn(1, 1, 200, 200)  # Crée un tenseur d'exemple avec la même taille que l'entrée
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        return x.view(1, -1).size(1)


model = CNN()

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Entraînement du modèle
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}  # Historique d'entraînement
for epoch in range(20):
    running_loss = 0.0
    running_accuracy = 0.0
    for i in range(0, len(X_train_tensor), 32):
        inputs, labels = X_train_tensor[i:i+32], y_train_tensor[i:i+32]

        # Remettre à zéro les gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        running_accuracy += (predicted == labels).sum().item() / len(y_train_tensor)
    print(f"Epoch {epoch+1},Run Accuracy: { running_accuracy}, Loss: {running_loss / len(X_train_tensor)}")
           
    # Évaluation sur l'ensemble de validation
    with torch.no_grad():
        outputs = model(X_val_tensor.unsqueeze(1))
        _, predicted = torch.max(outputs, 1)
        val_accuracy = (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
        val_loss = criterion(outputs, y_val_tensor)
        print(f'Validation Accuracy: {val_accuracy}, Validation Loss: {val_loss.item()}')
        history['val_accuracy'].append(val_accuracy)
        history['val_loss'].append(val_loss.item())

    # Enregistrer l'historique d'entraînement
    history['accuracy'].append((running_accuracy))
    history['loss'].append(running_loss)

# Affichage des courbes
# Accuracy
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()























#modèle CNN
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
history = model.fit(X_train, y_train, epochs=50, batch_size=96, validation_data=(X_val, y_val), verbose=1,
                    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

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










#fccnet + KD
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks, losses
from tensorflow.keras.applications import VGG16
import tensorflow.keras.backend as K
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

    # Conversion de l'image en niveaux de gris
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

# Chargement du modèle VGG16 entraîné
Asnet_model = models.load_model("VGG16.keras")

# Construction du modèle FCCNet avec distillation des connaissances
fccnet_input = layers.Input(shape=(200, 200, 2))
fccnet_conv1 = layers.Conv2D(32, (3, 3), activation='relu')(fccnet_input)
fccnet_pool1 = layers.MaxPooling2D((2, 2))(fccnet_conv1)
fccnet_conv2 = layers.Conv2D(64, (3, 3), activation='relu')(fccnet_pool1)
fccnet_pool2 = layers.MaxPooling2D((2, 2))(fccnet_conv2)
fccnet_conv3 = layers.Conv2D(128, (3, 3), activation='relu')(fccnet_pool2)
fccnet_pool3 = layers.MaxPooling2D((2, 2))(fccnet_conv3)
fccnet_conv4 = layers.Conv2D(128, (3, 3), activation='relu')(fccnet_pool3)
fccnet_pool4 = layers.MaxPooling2D((2, 2))(fccnet_conv4)
fccnet_flat = layers.Flatten()(fccnet_pool4)
fccnet_dense1 = layers.Dense(256, activation='relu')(fccnet_flat)
fccnet_dropout = layers.Dropout(0.5)(fccnet_dense1)
fccnet_output = layers.Dense(6, activation='softmax')(fccnet_dropout)

# Création du modèle FCCNet
fccnet_model = models.Model(inputs=fccnet_input, outputs=fccnet_output)

# Fonction de perte personnalisée avec distillation des connaissances
def knowledge_distillation_loss(y_true, y_pred, alpha=0.1, temperature=1):
    y_soft = K.softmax(y_true / temperature)
    y_pred_soft = K.softmax(y_pred / temperature)
    return alpha * losses.categorical_crossentropy(y_soft, y_pred_soft) + (1 - alpha) * losses.categorical_crossentropy(y_true, y_pred)

# Compilation du modèle FCCNet avec la fonction de perte personnalisée
fccnet_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                     loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, alpha=0.1, temperature=10),
                     metrics=['accuracy'])

# Entraînement du modèle FCCNet avec distillation des connaissances
history = fccnet_model.fit(X_train[:, :, :, :2], y_train, epochs=20, batch_size=32, validation_data=(X_val[:, :, :, :2], y_val), verbose=1,
                           callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# Évaluation du modèle FCCNet sur l'ensemble de test
test_loss, test_acc = fccnet_model.evaluate(X_test[:, :, :, :2], y_test)
print('Précision FCCNet sur l\'ensemble de test:', test_acc)

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














#Asnet model
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt

# Define your AssNet model
def AssNet(input_shape):
    model = models.Sequential([
    layers.Reshape((200, 200, 1), input_shape=(200, 200)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')
])

    return model

# Chargement des données
Datadir = "E:\\pcd\\NEU database\\NEU database"
target_size = (200, 200)

# Placeholder for augmented data
augmented_data = []
augmented_labels = []

# Mapping categories to numerical labels
category_to_label = {'Cr': 0, 'In': 1, 'Pa': 2, 'PS': 3, 'RS': 4, 'Sc': 5}

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
     # Conversion de l'image en niveaux de gris
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Application de la binarisation adaptative
    thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 5)


    # Extraire la catégorie à partir du nom de l'image
    category = img_name.split('_')[0]  # Extracting category from the image name
    if category not in category_to_label:
        print(f"Category '{category}' not found in category_to_label dictionary.")
        continue

    # Assigner une étiquette numérique à la catégorie
    label = category_to_label[category]

    # Ajouter les données augmentées et les étiquettes correspondantes
    augmented_data.append(gray_img)
    augmented_labels.append(label)

    # Ajouter les données binarisées
    augmented_data.append(thresh_img)
    augmented_labels.append(label)

# Convert lists to numpy arrays
augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)

# Division des données en ensembles d'entraînement (70%), de validation (15%) et de test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(augmented_data, augmented_labels, test_size=0.3, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# Construction du modèle AssNet
model = AssNet(input_shape=(200, 200, 3))

# Compilation du modèle
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle avec early stopping
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val), verbose=1,
                    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# Évaluation du modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Précision sur l\'ensemble de test:', test_acc)

model.save("Asnet.keras")

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




















#FCCNet
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import DepthwiseConv2D, Reshape
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import Lambda
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Activation, DepthwiseConv2D, Add, BatchNormalization
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

    # Conversion de l'image en niveaux de gris
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

# Chargement du modèle VGG16 entraîné
Asnet_model = models.load_model("Asnet.keras")


class ReshapeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.target_shape)

class TransposeLayer(Layer):
    def __init__(self, perm, **kwargs):
        super(TransposeLayer, self).__init__(**kwargs)
        self.perm = perm

    def call(self, inputs):
        return tf.transpose(inputs, self.perm)


def channel_shuffle(x, groups):
    batch_size, height, width, channels = x.shape
    channels_per_group = channels // groups
    x = Reshape((height, width, groups, channels_per_group))(x)
    x = TransposeLayer(perm=[0, 1, 2, 4, 3])(x)
    x = Reshape((height, width, channels))(x)
    return x

def shuffle_unit(inputs, groups):
    channels = inputs.shape[-1]
    channels_per_group = channels // groups
    x = Conv2D(channels // 2, (1, 1), activation=None)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Channel Shuffle
    x = channel_shuffle(x, groups)

    # Depthwise Convolution
    x = DepthwiseConv2D((3, 3), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(channels // 2, (1, 1), activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Effectuez une convolution supplémentaire pour adapter la forme des tenseurs
    residual_channels = channels - channels_per_group
    residual_path = Conv2D(residual_channels, (1, 1), activation=None)(inputs)
    residual_path = BatchNormalization()(residual_path)

    import keras.backend as K
    from keras.layers import ZeroPadding2D
    # Define a function to pad the last dimension
    def pad_last_dim(x):
     import tensorflow as tf
     return tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, channels - residual_channels]])

    
    # Define a Lambda layer to apply the padding
    output_shape = (200, 200, channels)

# Créer la couche Lambda avec la forme de sortie spécifiée
    pad_last_dim_layer = Lambda(pad_last_dim, output_shape=output_shape)

# Appliquer le rembourrage à residual_path
    residual_path = pad_last_dim_layer(residual_path)

    # Concaténer les chemins résiduels et principaux
    x = Add()([inputs, residual_path])
    x = Activation('relu')(x)

    # Votre code existant...

    return x


# Entrée du modèle
fccnet_input = Input(shape=(200, 200, 2))

# Convolution 3x3
x = Conv2D(32, (3, 3), activation='relu', padding='same')(fccnet_input)

# Couche Shuffle Unit
x = shuffle_unit(x, groups=2)

# Convolution 1x1
x = Conv2D(128, (1, 1), activation='relu')(x)

# Pooling adaptatif moyenne
x = GlobalAveragePooling2D()(x)

# Couche Dense
x = Dense(256, activation='relu')(x)

# Couche Dropout
x = Dropout(0.5)(x)

# Couche de sortie
fccnet_output = Dense(6, activation='softmax')(x)

# Création du modèle
fccnet_model = Model(inputs=fccnet_input, outputs=fccnet_output)


# Compilation du modèle FCCNet
fccnet_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                     loss={'output_classification': SparseCategoricalCrossentropy(from_logits=True),
                           'output_regularization': 'mean_squared_error'},
                     metrics={'output_classification': 'accuracy'},
                     loss_weights={'output_classification': 1.0, 'output_regularization': 0.5})

# Entraînement du modèle FCCNet avec distillation des connaissances
history = fccnet_model.fit(X_train[:, :, :, :2], {'output_classification': y_train, 'output_regularization': np.zeros_like(y_train)},
                           epochs=30, batch_size=64, validation_data=(X_val[:, :, :, :2], {'output_classification': y_val, 'output_regularization': np.zeros_like(y_val)}), 
                           verbose=1, callbacks=[callbacks.EarlyStopping(monitor='val_output_classification_loss', patience=5, restore_best_weights=True)])


# Évaluation du modèle FCCNet sur l'ensemble de test
test_loss, test_acc = fccnet_model.evaluate(X_test[:, :, :, :2], y_test)
print('Précision FCCNet sur l\'ensemble de test:', test_acc)

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

y_pred = np.argmax(fccnet_model.predict(X_test), axis=-1)

# Calculate additional metrics
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)







#VGG19
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications import VGG19
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Chargement des données
Datadir = "E:\\pcd\\NEU database\\NEU database"
target_size = (224, 224)  # VGG19 requires input images to be resized to (224, 224)

# Mapping categories to numerical labels
category_to_label = {'Cr': 0, 'In': 1, 'Pa': 2, 'PS': 3, 'RS': 4, 'Sc': 5}

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

    # Ajouter l'image originale et son étiquette
    category = img_name.split('_')[0]  # Extracting category from the image name
    if category in category_to_label:
        augmented_data.append(img)
        augmented_labels.append(category_to_label[category])
    else:
        print(f"Category '{category}' not found in category_to_label dictionary.")

    # Conversion de l'image en niveaux de gris
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Application de la binarisation adaptative
    thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 5)
    augmented_data.append(thresh_img)
    augmented_labels.append(category_to_label[category])

    

# Convert lists to numpy arrays
augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)


# Division des données en ensembles d'entraînement (70%), de validation (15%) et de test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(augmented_data, augmented_labels, test_size=0.3, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# Preprocess the data for VGG19
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)

# Load the pre-trained VGG19 model
base_model = VGG19(weights=None, include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
model = models.Sequential([
    base_model,
    layers.Flatten(),  # Flatten the output of the base model
    layers.Dense(256, activation='relu'),  # Add a dense layer with 256 units
    layers.Dropout(0.5),  # Add a dropout layer for regularization
    layers.Dense(6, activation='softmax')  # Add the final dense layer with 6 units for classification
])
# Compilation du modèle
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle avec early stopping
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_val, y_val), verbose=1,
                    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

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


y_pred = np.argmax(model.predict(X_test), axis=-1)

# Calculate additional metrics
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)
model.save("VGG.keras")








from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("VGG.keras")

# Load the images for testing
test_images = [...]  # List of paths to the test images

# Preprocess the test images
preprocessed_test_images = []
for img_path in test_images:
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = preprocess_input(img)
    preprocessed_test_images.append(img)

# Convert the list of images to numpy array
preprocessed_test_images = np.array(preprocessed_test_images)

# Make predictions
predictions = model.predict(preprocessed_test_images)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=-1)

# Print predicted labels
print("Predicted labels:", predicted_labels)

# Optionally, if you have ground truth labels for the test images
ground_truth_labels = [...]  # List of ground truth labels
ground_truth_labels = np.array(ground_truth_labels)

# Compare predicted labels with ground truth labels
accuracy = np.mean(predicted_labels == ground_truth_labels)
print("Accuracy:", accuracy)




