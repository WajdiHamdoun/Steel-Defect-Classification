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

























#training
import os
import cv2
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models

# Chargement des données
Datadir = "E:\\pcd\\NEU database\\NEU database"
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




pip install --upgrade tensorflow

