from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (Xception, VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2,
                                           ResNet152, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet,
                                           MobileNetV2, DenseNet121, DenseNet169, DenseNet201, NASNetMobile,
                                           NASNetLarge, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
                                           EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
                                           EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3,
                                           EfficientNetV2S, EfficientNetV2M, EfficientNetV2L, ConvNeXtTiny,
                                           ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from PyQt5.QtWidgets import QApplication, QFileDialog
from pathlib import Path

input_folder = None
output_folder = None
model_type = None
num_classes = 4

print("Initializing...")

# Create the application
app = QApplication([])

# Model selection dictionary
MODEL_SELECTION = {
    1: Xception,
    2: VGG16,
    3: VGG19,
    4: ResNet50,
    5: ResNet50V2,
    6: ResNet101,
    7: ResNet101V2,
    8: ResNet152,
    9: ResNet152V2,
    10: InceptionV3,
    11: InceptionResNetV2,
    12: MobileNet,
    13: MobileNetV2,
    14: DenseNet121,
    15: DenseNet169,
    16: DenseNet201,
    17: NASNetMobile,
    18: NASNetLarge,
    19: EfficientNetB0,
    20: EfficientNetB1,
    21: EfficientNetB2,
    22: EfficientNetB3,
    23: EfficientNetB4,
    24: EfficientNetB5,
    25: EfficientNetB6,
    26: EfficientNetB7,
    27: EfficientNetV2B0,
    28: EfficientNetV2B1,
    29: EfficientNetV2B2,
    30: EfficientNetV2B3,
    31: EfficientNetV2S,
    32: EfficientNetV2M,
    33: EfficientNetV2L,
    34: ConvNeXtTiny,
    35: ConvNeXtSmall,
    36: ConvNeXtBase,
    37: ConvNeXtLarge,
    38: ConvNeXtXLarge
}


def get_folder_path(message):
    while True:
        print(message)
        folder_path_input = QFileDialog().getExistingDirectory(None, message)

        if folder_path_input:
            folder_path = Path(folder_path_input)
            return folder_path
        else:
            print("Invalid input selected. Please try again.\n")


def invalid_input():
    print("Invalid input. Please try again.\n")


if model_type is None:
    while True:
        model_type = input("Which Scoring Model do you wanna use?\n1 = Xception\n2 = VGG16\n3 = VGG19\n4 = "
                           "ResNet50\n5 = ResNet50V2\n6 = ResNet101\n7 = ResNet101V2\n8 = ResNet152\n9 = "
                           "ResNet152V2\n10 = InceptionV3\n11 = InceptionResNetV2\n12 = MobileNet\13 = "
                           "MobileNetV2\n14 = DenseNet121\n15 = DenseNet169\n16 = DenseNet201\n17 = "
                           "NASNetMobile\n18 = NASNetLarge\n19 = EfficientNetB0\n20 = EfficientNetB1\n21 = "
                           "EfficientNetB2\n22 = EfficientNetB3\n23 = EfficientNetB4\n24 = EfficientNetB5\n25 = "
                           "EfficientNetB6\n26 = EfficientNetB7\n27 = EfficientNetV2B0\n28 = EfficientNetV2B1\n29 "
                           "= EfficientNetV2B2\n30 = EfficientNetV2B3\n31 = EfficientNetV2S\n32 = "
                           "EfficientNetV2M\n33 = EfficientNetV2L\n34 = ConvNeXtTiny\n35 = ConvNeXtSmall\n36 = "
                           "ConvNeXtBase\n37 = ConvNeXtLarge\n38 = ConvNeXtXLarge\nSelected Scoring Model: ")

        if model_type in [str(i) for i in range(1, 39)]:
            model_type = int(model_type)
            break
        invalid_input()

# Determine the input shape based on the selected model_type
if model_type in [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 19, 27, 34, 35, 36, 37, 38]:
    input_shape = (224, 224)
elif model_type in [20, 28]:
    input_shape = (240, 240)
elif model_type in [21, 29]:
    input_shape = (260, 260)
elif model_type in [1, 10, 11]:
    input_shape = (299, 299)
elif model_type in [22, 30]:
    input_shape = (300, 300)
elif model_type in [18]:
    input_shape = (331, 331)
elif model_type in [23]:
    input_shape = (380, 380)
elif model_type in [31]:
    input_shape = (384, 384)
elif model_type in [24]:
    input_shape = (456, 456)
elif model_type in [32, 33]:
    input_shape = (480, 480)
elif model_type in [25]:
    input_shape = (528, 528)
elif model_type in [26]:
    input_shape = (600, 600)
else:
    exit("Error 1")

# Define input directory
if input_folder is None:
    input_folder = get_folder_path("Choose your training data folder")

# Define input directory
if output_folder is None:
    output_folder = get_folder_path("Choose your model output folder")


# Define image dimensions and batch size
img_width, img_height = input_shape
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

print("Loading model: " + MODEL_SELECTION[int(model_type)])
# Load the pre-trained MobileNetV2 model (excluding the top layer)
base_model = MODEL_SELECTION[int(model_type)](weights='imagenet', include_top=False)

# Add custom layers for gender classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)  # Softmax for multi-class classification

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

# Extract base model name, number of classes, and total images
base_model_name = base_model.name
total_images = train_generator.samples

# Save the trained model with the specified information in the filename
model_filename = f'{base_model_name}_classes{num_classes}_images{total_images}.h5'
destination_file_path = output_folder / model_filename
model.save(destination_file_path)

print(f"Model saved in '{destination_file_path}'")

print("Gender training complete.")
exit()
