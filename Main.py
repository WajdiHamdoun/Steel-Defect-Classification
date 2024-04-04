import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Définition de la classe du dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Chargement des données
Datadir = "E:\\pcd\\NEU database\\NEU database"
target_size = (224, 224)  # VGG19 requires input images to be resized to (224, 224)

# Mapping categories to numerical labels
category_to_label = {'Cr': 0, 'In': 1, 'Pa': 2, 'PS': 3, 'RS': 4, 'Sc': 5}

# Charger les données et les étiquettes
augmented_data = []
augmented_labels = []

for img_name in os.listdir(Datadir):
    img_path = os.path.join(Datadir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        continue
    img = cv2.resize(img, target_size)
    category = img_name.split('_')[0]
    if category in category_to_label:
        augmented_data.append(img)
        augmented_labels.append(category_to_label[category])
    else:
        print(f"Category '{category}' not found in category_to_label dictionary.")
        continue
    img_horizontal = cv2.flip(img, 1)
    augmented_data.append(img_horizontal)
    augmented_labels.append(category_to_label[category])
    img_vertical = cv2.flip(img, 0)
    augmented_data.append(img_vertical)
    augmented_labels.append(category_to_label[category])

augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)

# Division des données en ensembles d'entraînement (70%), de validation (15%) et de test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(augmented_data, augmented_labels, test_size=0.3, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# Transformations d'images
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Création des datasets et dataloaders
train_dataset = CustomDataset(X_train, y_train, transform=data_transform)
val_dataset = CustomDataset(X_val, y_val, transform=data_transform)
test_dataset = CustomDataset(X_test, y_test, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Chargement du modèle pré-entraîné VGG19
model = models.vgg19(pretrained=True)

# Remplacement de la dernière couche entièrement connectée
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, 256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 6)])  # Add custom layers
model.classifier = nn.Sequential(*features)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Entraînement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
num_epochs = 15
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, '
          f'Val Loss: {val_loss:.4f}, '
          f'Val Accuracy: {val_accuracy:.4f}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# Évaluation du modèle sur l'ensemble de test
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print('Accuracy on test set:', test_accuracy)

# Calcul des prédictions et du rapport de classification
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

report = classification_report(y_true, y_pred)
print('Classification Report:\n', report)

torch.save(model.state_dict(), 'best_model.pth')








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
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Chargement des données
Datadir = "C:/Users/pc/Desktop/NEU surface defect database"
target_size = (224, 224)  # VGG19 requires input images to be resized to (224, 224)

# Mapping categories to numerical labels
category_to_label = {'Cr': 0, 'In': 1, 'Pa': 2, 'PS': 3, 'RS': 4, 'Sc': 5}


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

    # Flip the image horizontally
    img_horizontal = cv2.flip(img, 1)
    
    # Ajouter l'image horizontalement flipée et sa même étiquette
    augmented_data.append(img_horizontal)
    augmented_labels.append(category_to_label[category])
    
    # Flip the image vertically
    img_vertical = cv2.flip(img, 0)
    
    # Ajouter l'image verticalement flipée et sa même étiquette
    augmented_data.append(img_vertical)
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
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False
# Create a new Sequential model
model = models.Sequential()

# Add the pre-trained VGG19 model to the new model
model.add(base_model)

# Add the Flatten layer explicitly
model.add(layers.Flatten())

# Add custom classification layers on top of the Flatten layer
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(6, activation='softmax'))


# Compilation du modèle
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle avec early stopping
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_val, y_val), verbose=1,
                    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

model.save_weights("VGG.weights.h5")
json_string = model.to_json()
with open("vgg.json", "w") as f:
    f.write(json_string)
model.save("VGG.keras")
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




















import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import json

# Charger le modèle
with open("C:\\Users\\wajdi\\Documents\\GitHub\\PCD\\CNN.json", "r") as json_file:
    loaded = json_file.read()
model = tf.keras.models.model_from_json(loaded)
model.load_weights("CNN.h5")

# Chemin vers le dossier contenant les images de test
test_folder = "C:\\Users\\wajdi\\Desktop\\images"
output_folder = "C:\\Users\\wajdi\\Desktop\\images2"
target_size = (200, 200)

# Liste pour stocker les étiquettes prédites
predicted_labels = []

# Boucle à travers chaque image dans le dossier de test
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    
    # Lire et prétraiter l'image
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)  # Redimensionner l'image pour correspondre à la taille d'entrée du modèle
    
    # Augmentation de contraste
    enhanced_img = cv2.convertScaleAbs(img, alpha=1.0, beta=0)
    
    # Conversion en niveaux de gris
    gray_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    
    # Application de la binarisation adaptative
    _, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    zeros_img = np.zeros_like(thresh_img)

    # Stack the images together along the channel axis
    stacked_img = np.stack([gray_img, zeros_img], axis=-1)
    

    # Ajouter une dimension de lot
    stacked_img = np.expand_dims(stacked_img, axis=0)
    plt.imshow(stacked_img[0, ..., 0], cmap='gray')  # Affiche la première canal en niveaux de gris

    plt.axis('off')  # Désactive les axes
    plt.show()

        
    # Prédire les probabilités de classe
    predictions = model.predict(stacked_img)
    
    # Obtenir l'étiquette prédite
    predicted_label = np.argmax(predictions)
    

    # Map the predicted label back to category
    label_to_category = {0: 'Cr', 1: 'In', 2: 'Pa', 3: 'PS', 4: 'RS', 5: 'Sc'}
    predicted_category = label_to_category[predicted_label]
    
    # Ajouter l'étiquette prédite à la liste
    predicted_labels.append(predicted_category)

# Afficher les étiquettes prédites
for img_name, predicted_label in zip(os.listdir(test_folder), predicted_labels):
    print(f"Image: {img_name}, Predicted Label: {predicted_label}")
























import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input
with open("C:\\Users\\wajdi\\Documents\\GitHub\\PCD\\vgg.json", "r") as json_file:
    loaded = json_file.read()
model = tf.keras.models.model_from_json(loaded)
model.load_weights("VGG.weights.h5")


# Path to the folder containing test images
test_folder = "C:\\Users\\wajdi\\Desktop\\images"

# List to store predicted labels
predicted_labels = []

# Loop through each image in the test folder
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    
    # Read and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match the input size of VGG19
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict the class probabilities
    predictions = model.predict(img)
    
    # Get the predicted label
    predicted_label = np.argmax(predictions)
    
    # Map the predicted label back to category
    label_to_category = {0: 'Cr', 1: 'In', 2: 'Pa', 3: 'PS', 4: 'RS', 5: 'Sc'}
    predicted_category = label_to_category[predicted_label]
    
    # Append the predicted label to the list
    predicted_labels.append(predicted_category)

# Display the predicted labels
for img_name, predicted_label in zip(os.listdir(test_folder), predicted_labels):
    print(f"Image: {img_name}, Predicted Label: {predicted_label}")