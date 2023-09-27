from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from PyQt5.QtWidgets import QApplication, QFileDialog
from pathlib import Path

input_folder = None

print("Initializing...")

# Create the application
app = QApplication([])


def get_folder_path(message):
    while True:
        print(message)
        folder_path_input = QFileDialog().getExistingDirectory(None, message)

        if folder_path_input:
            folder_path = Path(folder_path_input)
            return folder_path
        else:
            print("Invalid input selected. Please try again.\n")


# Define input directory
if input_folder is None:
    input_folder = get_folder_path("Choose your input folder")


# Define image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Create a data generator for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    input_folder,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # Multi-class classification (male, female, both, neither)
    shuffle=True
)

print("Loading model...")
# Load the pre-trained MobileNetV2 model (excluding the top layer)
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Add custom layers for gender classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # Softmax for multi-class classification

# Create the custom model
model = Model(inputs=base_model.input, outputs=predictions)

print("Model loaded")

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 10  # Adjust as needed
history = model.fit(train_generator,
                    epochs=epochs)

# Save the trained model
model.save('gender_classification_model.h5')

print("Gender training complete.")
exit()
