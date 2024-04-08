

# preproc

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

Datadir = "C:/Users/pc/Desktop/NEU surface defect database"
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
    flipped_img_horizontal = cv2.flip(enhanced_img, 1)
    
    # Flip the image vertically
    flipped_img_vertical = cv2.flip(enhanced_img, 0)
    
    # Append original image, flipped images, and thresholded image to the list
    augmented_data.extend([enhanced_img, flipped_img_horizontal, flipped_img_vertical])

# Convert the list to a numpy array
augmented_data = np.array(augmented_data)

# Display some of the augmented images
fig, axes = plt.subplots(5, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(cv2.cvtColor(augmented_data[i], cv2.COLOR_BGR2RGB))
    ax.axis('off')
    if i % 3 == 0 :
        ax.set_title(f'Original Image {(i//3)+1}')
    elif i % 2 == 0:
        ax.set_title(f'Rotated Image horizontal {(i//3)+1}')
    else :
        ax.set_title(f'Rotated Image vertical {(i//3)+1}')
plt.show()

num_images = len(augmented_data)
print("Number of images in augmented dataset:", num_images)
















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

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')
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
model.save_weights("VGG.weights.h5")












#Testing
import numpy as np
import cv2
import os
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg19 import preprocess_input

# Recreate the model architecture
base_model1 = VGG19(weights=None, include_top=False, input_shape=(224, 224, 3))
for layer in base_model1.layers:
    layer.trainable = False

model1 = models.Sequential([
    base_model1,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')
])

# Make sure the model is built by calling it with a sample input
# This step ensures that all layers are in a built state
_ = model1(np.random.random((1, 224, 224, 3)))

# Load the weights
model1.load_weights('VGG.weights.h5')


# Path to the folder containing test images
test_folder = "C:/Users/pc/Desktop/DS_Test"

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
    predictions = model1.predict(img)
    
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
    




















#import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
import os
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg19 import preprocess_input
import tkinter as tk
import numpy as np
import cv2
import os
from PIL import Image, ImageTk
from tensorflow.keras import layers, models
from collections import Counter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import urllib.request

# Assuming you have defined the model and other necessary imports
base_model1 = VGG19(weights=None, include_top=False, input_shape=(224, 224, 3))
for layer in base_model1.layers:
    layer.trainable = False
model1 = models.Sequential([
    base_model1,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')
])
_ = model1(np.random.random((1, 224, 224, 3)))
model1.load_weights('VGG.weights.h5')
class Application(tk.Tk):
    # Define constants for paragraphs of each defect
    PARAGRAPHS = {
        'Crack': "A crack is a line on the surface of something along which it has split without breaking apart.",
        'Incomplete Penetration': "Incomplete penetration occurs when the weld metal does not extend into the root of the weld joint.",
        'Porosity': "Porosity is the presence of small voids or cavities in the weld metal caused by gas entrapment during solidification.",
        'Puddle Spatter': "Puddle spatter is small particles of molten metal expelled during welding.",
        'Root Smutting': "Root smutting is the presence of sooty deposits or contamination at the root of the weld joint.",
        'Slag': "Slag is the non-metallic byproduct of the welding process that forms a layer on the surface of the weld bead."
    }

    def __init__(self):
        super().__init__()
        self.title('Steel Defect Classification')
        self.geometry("800x600")
        self.configure(background='#434547')  # Set the main window background
        
        # Custom styles for ttk widgets
        self.style = ttk.Style(self)
        self.style.configure('TLabel', font=('Arial', 12), foreground='#FFFFFF', background='#434547')
        self.style.configure('TButton', font=('Arial', 12), background='#5C5F63', foreground='#FFFFFF')
        self.style.map('TButton', background=[('active', '#848484')])  # Change background color on button click
        self.initialize_ui()
        
    def initialize_ui(self):
        self.clear_widgets()
        self.auth_frame = tk.Frame(self, background='#434547')  # Using tk.Frame to allow background color
        self.auth_frame.pack(padx=10, pady=10, fill='x', expand=True)
        
        # ttk widgets like labels and buttons will inherit the style specified but won't directly change their background
        self.password_label = ttk.Label(self.auth_frame, text="Enter Password:")
        self.password_label.pack(side='left')
        
        self.password_entry = ttk.Entry(self.auth_frame)  # Entry widget doesn't support ttk in standard Tkinter
        self.password_entry.pack(side='left', padx=10)
        
        self.login_button = ttk.Button(self.auth_frame, text="Login", command=self.authenticate)
        self.login_button.pack(side='left')


    def authenticate(self):
        password = self.password_entry.get()
        self.isAdmin = False
        if password == "user":
            self.user_interface()
        elif password == "admin":
            self.isAdmin = True
            self.admin_interface()
        else:
            messagebox.showerror("Error", "Incorrect Password")

    def user_interface(self):
        self.clear_widgets()
        main_frame = tk.Frame(self, background='#434547')  # Using tk.Frame to allow background color
        main_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        self.upload_button = ttk.Button(main_frame, text="Upload and Classify Folder", command=self.upload_folder)
        self.upload_button.pack()

        self.back_button = ttk.Button(main_frame, text="Go Back", command=self.initialize_ui)
        self.back_button.pack(pady=10)


    def admin_interface(self):
        self.clear_widgets()
        main_frame = tk.Frame(self, background='#434547')  # Using tk.Frame to allow background color
        main_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Add background picture
        try:
            # Load the image from a local file
            image_path = "wall.jpg"  # Specify the correct file path
            pil_image = Image.open(image_path)
            width, height = self.winfo_screenwidth(), self.winfo_screenheight()
            pil_image = pil_image.resize((width, height))
            tk_image = ImageTk.PhotoImage(pil_image)
            background_label = tk.Label(main_frame, image=tk_image)
            background_label.image = tk_image  # Keep a reference to prevent garbage collection
            background_label.place(relwidth=1, relheight=1)
        except Exception as e:
            print("Failed to load background image:", e)

        # Create a dropdown menu for defect classification
        defect_label = ttk.Label(main_frame, text="Select Defect Type:")
        defect_label.pack()

        defect_options = ['Rolling Scale', 'Plaque', 'Pitting Surface ', 'Inclusions', 'Scratches']
        self.defect_var = tk.StringVar()
        self.defect_var.set(defect_options[0])  # Set default value
        defect_dropdown = ttk.OptionMenu(main_frame, self.defect_var, *defect_options)
        defect_dropdown.pack()
        
        # Text area for admin to provide paragraphs for each defect
        paragraph_label = ttk.Label(main_frame, text="Paragraph for the Selected Defect:")
        paragraph_label.pack()

        selected_defect = self.defect_var.get()
        paragraph_text = self.PARAGRAPHS.get(selected_defect, "Paragraph not found.")
        paragraph_display = tk.Text(main_frame, height=10)
        paragraph_display.insert(tk.END, paragraph_text)
        paragraph_display.config(state=tk.DISABLED)  # Make text area read-only
        paragraph_display.pack()

        # Button to submit defect classification and paragraph
        submit_button = ttk.Button(main_frame, text="Submit", command=self.submit_defect_classification)
        submit_button.pack()

        # Button to go back
        self.back_button = ttk.Button(main_frame, text="Go Back", command=self.initialize_ui)
        self.back_button.pack(pady=10)
        # Button to select the input folder
        select_folder_button = ttk.Button(main_frame, text="Select Input Image Folder", command=self.select_input_folder)
        select_folder_button.pack()

    def select_input_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            messagebox.showinfo("Input Image Folder Selected", f"Selected folder: {folder_path}")

    def submit_defect_classification(self):
        selected_defect = self.defect_var.get()
        paragraph = self.paragraph_entry.get("1.0", tk.END)  # Get the paragraph from the text area

        # Save the paragraph to a file
        with open(f"{selected_defect}_paragraph.txt", "w") as file:
            file.write(paragraph)

        # Classify images from the folder provided by the admin
        folder_path = filedialog.askdirectory()
        if folder_path:
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Filter for image files
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (224, 224))
                    image = preprocess_input(image)
                    image = np.expand_dims(image, axis=0)

                    predictions = model1.predict(image)
                    predicted_class = np.argmax(predictions)
                    categories = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
                    predicted_category = categories[predicted_class]

                    # Create a folder for each defect type if it doesn't exist
                    defect_folder = f"{selected_defect}_images"
                    os.makedirs(defect_folder, exist_ok=True)

                    # Save the image to the corresponding defect folder
                    cv2.imwrite(os.path.join(defect_folder, img_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Show a confirmation message
        messagebox.showinfo("Success", f"Defect '{selected_defect}' classified successfully!\nParagraph saved to '{selected_defect}_paragraph.txt'")

    def display_results(self, classifications):
        results_window = tk.Toplevel(self)
        results_window.title("Classification Results")
        results_window.geometry("600x400")
        
        text_area = tk.Text(results_window)
        text_area.pack(side='left', fill='both', expand=True)
        
        for classification in classifications:
            text_area.insert(tk.END, classification + "\n")
        
        if self.isAdmin:  # Show detailed analytics for admins only
            self.show_pie_chart(classifications, results_window)

    def show_pie_chart(self, classifications, parent_window):
        counts = Counter([x.split(': ')[1] for x in classifications])
        categories = list(counts.keys())
        sizes = list(counts.values())
        
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=categories, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
        ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        
        canvas = FigureCanvasTkAgg(fig, master=parent_window)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side='right', fill='both', expand=True)

    def clear_widgets(self):
        for widget in self.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    app = Application()
    app.mainloop()
