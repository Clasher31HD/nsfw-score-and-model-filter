import tensorflow as tf
from tensorflow.keras.applications import (DenseNet121, DenseNet169, InceptionV3, MobileNetV2,
                                           ResNet50, ResNet101, ResNet152, EfficientNetB0, EfficientNetB7)
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np


def get_image_score(image_path, model):
    try:
        # Load the chosen model
        model = model(weights='imagenet')

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=model.input_shape[1:3])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        if model.__class__.__name__ == 'ResNet50':
            preprocessed_img = resnet_preprocess_input(img_array.copy())
        elif model.__class__.__name__ == 'DenseNet121':
            preprocessed_img = densenet_preprocess_input(img_array.copy())
        elif model.__class__.__name__ == 'DenseNet169':
            preprocessed_img = densenet_preprocess_input(img_array.copy())
        elif model.__class__.__name__ == 'InceptionV3':
            preprocessed_img = inceptionv3_preprocess_input(img_array.copy())
        elif model.__class__.__name__ == 'MobileNetV2':
            preprocessed_img = mobilenetv2_preprocess_input(img_array.copy())
        else:
            preprocessed_img = model.preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(preprocessed_img)
        decoded_predictions = model.decode_predictions(predictions, top=3)[0]

        # Print the top predictions
        for _, label, score in decoded_predictions:
            print(f"{label}: {score}")
    except Exception as e:
        print(f"Error processing image: {e}")


# Provide the path to your image
image_path = "C:/Users/I539356/Downloads/www.freepik.com/All/attractive-blonde-female-relaxing-couch-room-with-books-stan.jpg"

# Model selection dictionary
model_selection = {
    1: InceptionV3,
    2: DenseNet121,
    3: DenseNet169,
    4: MobileNetV2,
    5: ResNet50,
    6: ResNet101,
    7: ResNet152,
    8: EfficientNetB0,
    9: EfficientNetB7
}

# Choose a model from the options
chosen_model_input = input("Enter the number corresponding to the desired model:"
                           "\n1 = InceptionV3, 2 = DenseNet121, 3 = DenseNet169, 4 = MobileNetV2, "
                           "5 = ResNet50, 6 = ResNet101, 7 = ResNet152, 8 = EfficientNetB0, 9 = EfficientNetB7: ")

try:
    chosen_model_input = int(chosen_model_input)
    if chosen_model_input in model_selection:
        chosen_model = model_selection[chosen_model_input]
        # Calculate the image score using the chosen model
        get_image_score(image_path, chosen_model)
    else:
        print("Invalid model selection.")
except ValueError:
    print("Invalid input. Please enter a valid model number.")
