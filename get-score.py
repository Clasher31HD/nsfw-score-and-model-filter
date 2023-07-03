import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3, preprocess_input
from tensorflow.keras.preprocessing import image


def get_image_score(image_path):
    # Load and preprocess the image
    if model_type == "1" or model_type == "2":
        img = image.load_img(image_path, target_size=(224, 224))
    else:
        img = image.load_img(image_path, target_size=(300, 300))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Use the pre-trained model to predict the image class probabilities
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Return the predicted class index and corresponding score
    return predicted_class, predictions[0][predicted_class]


model_type = input("Which Scoring Model do you wanna use?")
if model_type == "1":
    # Load the pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')
elif model_type == "2":
    model = ResNet101(weights='imagenet')
else:
    model = EfficientNetV2B3(weights='imagenet')


# Provide the path to the input image
input_image_path = "C:/Users/I539356/Downloads/1859647-831502033.jpg"

# Get the score for the input image
class_index, score = get_image_score(input_image_path)

# Print the class index and score
print(f"Image: {input_image_path}")
print(f"Class index: {class_index}")
print(f"Score: {score}")
