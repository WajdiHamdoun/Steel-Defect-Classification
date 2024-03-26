import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

def load_data(datadir, target_size):
    augmented_data = []
    augmented_labels = []
    
    category_to_label = {'Cr': 0, 'In': 1, 'Pa': 2, 'PS': 3, 'RS': 4, 'Sc': 5}
    
    for img_name in os.listdir(datadir):
        img_path = os.path.join(datadir, img_name)
    
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_path}")
            continue
    
        img = cv2.resize(img, target_size)
    
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 5)
    
        category = img_name.split('_')[0]
        if category not in category_to_label:
            print(f"Category '{category}' not found in category_to_label dictionary.")
            continue
    
        label = category_to_label[category]
    
        stacked_img = np.stack([gray_img, thresh_img, gray_img], axis=-1)
    
        augmented_data.append(stacked_img)
        augmented_labels.append(label)
    
    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)
    
    return augmented_data, augmented_labels

def build_model(input_shape):
    base_vgg16 = VGG16(weights=None, include_top=False, input_shape=input_shape)
    
    x = layers.Flatten()(base_vgg16.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(6, activation='softmax')(x)
    
    vgg16_model = models.Model(inputs=base_vgg16.input, outputs=output)
    
    return vgg16_model

def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1,
                          callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    
    return history

def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Précision sur l\'ensemble de test:', test_acc)

def plot_history(history):
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

if __name__ == "__main__":
    Datadir = "E:\\pcd\\NEU database\\NEU database"
    target_size = (200, 200)
    
    augmented_data, augmented_labels = load_data(Datadir, target_size)
    
    X_train, X_temp, y_train, y_temp = train_test_split(augmented_data, augmented_labels, test_size=0.3, random_state=20)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)
    
    vgg16_model = build_model(input_shape=(200, 200, 3))
    
    history = train_model(vgg16_model, X_train, y_train, X_val, y_val)
    
    vgg16_model.save("VGG16_trained.keras")
    
    evaluate_model(vgg16_model, X_test, y_test)
    
    plot_history(history)
