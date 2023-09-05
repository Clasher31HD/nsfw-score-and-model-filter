import shutil
from pathlib import Path
import PIL
import json
from PIL import Image
import re
import numpy as np
import opennsfw2 as n2
from tensorflow.keras.applications import (Xception, VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2,
                                           ResNet152, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet,
                                           MobileNetV2, DenseNet121, DenseNet169, DenseNet201, NASNetMobile,
                                           NASNetLarge, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
                                           EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
                                           EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3,
                                           EfficientNetV2S, EfficientNetV2M, EfficientNetV2L, ConvNeXtTiny,
                                           ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge)
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnetv2_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess_input
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess_input
from tensorflow.keras.preprocessing import image
from PyQt5.QtWidgets import QApplication, QFileDialog

# Config path
config_path = Path("config.json")

# Config values
autonomous = None
input_folder = None
output_folder = None
move_or_copy = None
mode = None
parameter_list = None
experimental = None
own_parameters = None
parameters = None
strict_parameters = None
split_words = None
model_type = None
score_or_class = None

print("Initializing...")

# Create the application
app = QApplication([])

# Initialize variables
nsfw_folder_name = None
score_folder_name = None
model_folder_name = None
score_range_type = None

# Define the regular expression pattern for invalid characters
invalid_chars_pattern = r'[<>:"-_/\\|?*().;#{}[\]\n]'

# Constants for NSFW ranges
NSFW_RANGES = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 0.9),
    (0.9, 0.95),
    (0.95, 0.99),
    (0.99, 0.995),
    (0.995, 1.0)
]

# Constants for score ranges
SCORE_RANGES_SMALL = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0)
]

SCORE_RANGES_BIG = [
    (0, 2),
    (2, 4),
    (4, 6),
    (6, 8),
    (8, 10)
]

SMALL_SCORE_MODELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                      28, 29, 30, 31, 32, 33]
BIG_SCORE_MODELS = [34, 35, 36, 37, 38]

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


# Function to check the range for a value and return the corresponding folder name
def get_folder_name(value, ranges):
    if value is None:
        return 'None'
    for i, (start, end) in enumerate(ranges):
        if start <= value < end:
            return f"{start}-{end}"
    return 'None'


def is_nsfw(image_path):
    try:
        # Load image and resize to maximum of 512 pixels
        img = Image.open(image_path)
        img.thumbnail((512, 512))

        # Check NSFW probability using the NSFW detector
        nsfw_probability = n2.predict_image(image_path)
        return nsfw_probability
    except (PIL.UnidentifiedImageError, OSError) as e:
        print(f"Skipping image '{image_path.name}' due to an error: {str(e)}")
        return None


def get_image_score(image_path):
    # Load and preprocess the image
    if model_type in [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 19, 27, 34, 35, 36, 37, 38]:
        img = image.load_img(image_path, target_size=(224, 224))
    elif model_type in [20, 28]:
        img = image.load_img(image_path, target_size=(240, 240))
    elif model_type in [21, 29]:
        img = image.load_img(image_path, target_size=(260, 260))
    elif model_type in [1, 10, 11]:
        img = image.load_img(image_path, target_size=(299, 299))
    elif model_type in [22, 30]:
        img = image.load_img(image_path, target_size=(300, 300))
    elif model_type in [18]:
        img = image.load_img(image_path, target_size=(331, 331))
    elif model_type in [23]:
        img = image.load_img(image_path, target_size=(380, 380))
    elif model_type in [31]:
        img = image.load_img(image_path, target_size=(384, 384))
    elif model_type in [24]:
        img = image.load_img(image_path, target_size=(456, 456))
    elif model_type in [32, 33]:
        img = image.load_img(image_path, target_size=(480, 480))
    elif model_type in [25]:
        img = image.load_img(image_path, target_size=(528, 528))
    elif model_type in [26]:
        img = image.load_img(image_path, target_size=(600, 600))
    else:
        exit("Error 1")

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    if model_type in [1]:
        img = xception_preprocess_input(img)
    elif model_type in [2]:
        img = vgg16_preprocess_input(img)
    elif model_type in [3]:
        img = vgg19_preprocess_input(img)
    elif model_type in [4, 6, 8]:
        img = resnet_preprocess_input(img)
    elif model_type in [5, 7, 9]:
        img = resnetv2_preprocess_input(img)
    elif model_type in [10]:
        img = inceptionv3_preprocess_input(img)
    elif model_type in [11]:
        img = inception_resnetv2_preprocess_input(img)
    elif model_type in [12]:
        img = mobilenet_preprocess_input(img)
    elif model_type in [13]:
        img = mobilenetv2_preprocess_input(img)
    elif model_type in [14, 15, 16]:
        img = densenet_preprocess_input(img)
    elif model_type in [17, 18]:
        img = nasnet_preprocess_input(img)
    elif model_type in range(19, 26):
        img = efficientnet_preprocess_input(img)
    elif model_type in range(27, 33):
        img = efficientnetv2_preprocess_input(img)
    elif model_type in range(34, 39):
        img = convnext_preprocess_input(img)
    else:
        exit("Error 2")

    # Use the pre-trained model to predict the image class probabilities
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Return the predicted class index and corresponding score
    return predicted_class, predictions[0][predicted_class]


def extract_model_name(file_path):
    # Extract the model name from the metadata dictionary
    image = Image.open(file_path)
    metadata = image.info
    params = metadata.get("parameters", "")
    model_start = params.find("Model:") + len("Model:")
    model_end = params.find(",", model_start)
    model_name = params[model_start:model_end].strip()
    return model_name


def extract_parameters(file_path):
    image = Image.open(file_path)
    metadata = image.info
    params = metadata.get("parameters", "")
    if "Negative prompt:" in params:
        result = params.split("Negative prompt:", 1)[0].strip()
    elif "Steps:" in params:
        result = params.split("Steps:", 1)[0].strip()
    else:
        exit("Error 6")
    cleaned_result = re.sub(invalid_chars_pattern, '', result)

    if ',' not in cleaned_result or split_words:
        # Split the cleaned folder name by whitespaces and remove invalid characters
        removed_commas = re.sub(r',', '', cleaned_result)
        separate_list = re.split(r'\s+', removed_commas)
    else:
        # Split the cleaned folder name by commas and remove invalid characters
        separate_list = re.split(r',', cleaned_result)

    parameter_list = [item.strip() for item in separate_list if item.strip() and len(item.strip()) <= 100]
    return parameter_list


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


def config_key_not_exists(key):
    print(f"The '{key}' key does not exist in the config file.")


def invalid_config(key):
    print(f"Config value of '{key}' is invalid")


def check_config():
    print("Checking for config...")

    if config_path.exists():
        print("Config file found")
        print("Loading config values")
        # Open and read the config file
        with open(config_path, 'r') as config_file:
            config_data = json.load(config_file)

        keys_to_check = ["autonomous", "mode", "input_folder", "output_folder", "move_or_copy"]

        for key in keys_to_check:
            if key not in config_data:
                config_key_not_exists(key)
                return False

        autonomous = config_data["autonomous"]
        mode = config_data["mode"]
        input_folder = Path(config_data["input_folder"])
        output_folder = Path(config_data["output_folder"])
        move_or_copy = config_data["move_or_copy"]

        if autonomous not in ["True", "False"]:
            invalid_config("autonomous")
            return False
        if mode not in [i for i in range(1, 17)]:
            invalid_config("mode")
            return False
        if not input_folder.exists() or not input_folder.is_dir():
            invalid_config("input_folder")
            return False
        if not output_folder.exists() or not output_folder.is_dir():
            invalid_config("output_folder")
            return False
        if move_or_copy not in [1, 2]:
            invalid_config("move_or_copy")
            return False

        if mode in [2, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15]:
            if "model_type" not in config_data:
                config_key_not_exists("model_type")
                return False
            if "score_or_class" not in config_data:
                config_key_not_exists("score_or_class")
                return False

            model_type = config_data["model_type"]
            score_or_class = config_data["score_or_class"]

            if model_type not in [i for i in range(1, 39)]:
                invalid_config("model_type")
                return False
            if score_or_class not in ["s", "c"]:
                invalid_config("score_or_class")
                return False

        if mode == 16:  # Fixed the condition
            if "experimental" not in config_data:
                config_key_not_exists("experimental")
                return False
            else:
                if "own_parameters" not in config_data:
                    config_key_not_exists("own_parameters")
                    return False
                if "split_words" not in config_data:
                    config_key_not_exists("split_words")
                    return False

                experimental = config_data["experimental"]
                own_parameters = config_data["own_parameters"]
                split_words = config_data["split_words"]

                if experimental not in ["y", "n"]:
                    invalid_config("experimental")
                    return False
                if own_parameters not in ["y", "n"]:
                    invalid_config("own_parameters")
                    return False
                if split_words not in ["y", "n"]:
                    invalid_config("split_words")
                    return False

                if own_parameters == "y":
                    if "parameters" not in config_data:
                        config_key_not_exists("parameters")
                        return False
                    if "strict_parameters" not in config_data:
                        config_key_not_exists("strict_parameters")
                        return False

                    parameters = config_data["parameters"]
                    strict_parameters = config_data["strict_parameters"]

                    if parameters is None:
                        invalid_config("parameters")
                        return False
                    if strict_parameters not in ["y", "n"]:
                        invalid_config("strict_parameters")
                        return False
            return True

        else:
            print("No config file found, proceeding without...")
            return False


while True:
    if check_config():
        print("Config successfully checked.")
        print("Loading config...")

        with open(config_path, 'r') as config_file:
            config_data = json.load(config_file)

            autonomous = config_data["autonomous"]
            mode = config_data["mode"]
            input_folder = Path(config_data["input_folder"])
            output_folder = Path(config_data["output_folder"])
            move_or_copy = config_data["move_or_copy"]

            if mode in [2, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15]:
                model_type = config_data["model_type"]
                score_or_class = config_data["score_or_class"]

            if mode == 16:
                experimental = config_data["experimental"]
                own_parameters = config_data["own_parameters"]
                split_words = config_data["split_words"]

                if own_parameters == "y":
                    parameters = config_data["parameters"]
                    strict_parameters = config_data["strict_parameters"]

        print("Config successfully loaded.")
        break
    else:
        if mode is None:
            while True:
                mode = input("Enter Mode:\n1 = NSFW\n2 = Score\n3 = Model\n4 = NSFW/Model\n5 = NSFW/Score\n6 = "
                             "Score/NSFW\n7 = Score/Model\n8 = Model/NSFW\n9 = Model/Score\n10 = NSFW/Score/Model\n11 "
                             "= NSFW/Model/Score\n12 = Score/NSFW/Model\n13 = Score/Model/NSFW\n14 = "
                             "Model/NSFW/Score\n15 = Model/Score/NSFW\n16 = Parameter (Experimental)\nSelected Mode: ")

                if mode in [str(i) for i in range(1, 16)]:
                    mode = int(mode)
                    break
                elif mode == "16":
                    mode = int(mode)
                    if experimental is None:
                        while True:
                            experimental = input("This mode is experimental and creates a lot of duplicate "
                                                 "files.\nAre you sure you want to continue? (y = yes, n = no): ")
                            if experimental == "y" or experimental == "n":
                                break
                            invalid_input()

                    if own_parameters is None:
                        while True:
                            own_parameters_input = input("Do you wanna filter by own parameters? (y = yes, n = no): ")
                            if own_parameters_input == "y":
                                own_parameters = input("Type your filter parameters separated by commas (,): ")
                                own_parameters = own_parameters.replace(" ", "")  # Remove any spaces in the input
                                parameters = own_parameters.split(",")  # Split the input at each comma

                                if strict_parameters is None:
                                    while True:
                                        strict_parameters_input = input("Do you wanna filter by strictly all "
                                                                        "parameters? (y = yes, n = no): ")
                                        if strict_parameters_input == "y":
                                            strict_parameters = True
                                            break
                                        elif strict_parameters_input == "n":
                                            strict_parameters = False
                                            break
                                        invalid_input()
                                break
                            elif own_parameters_input == "n":
                                parameters = None
                                break
                            invalid_input()
                    if split_words is None:
                        while True:
                            split_words_input = input("Do you wanna split each word from existing images? (y = yes, "
                                                      "n = no): ")
                            if split_words_input == "y":
                                split_words = True
                                break
                            elif split_words_input == "n":
                                split_words = False
                                break
                            invalid_input()
                        break
                invalid_input()
            break

if mode in [2, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15]:
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

    if score_or_class is None:
        while True:
            score_or_class = input("Do you wanna filter by score or class? (s = score, c = class) ")
            if score_or_class == "s" or score_or_class == "c":
                break
            invalid_input()

    print("Loading scoring model...")
    model = MODEL_SELECTION[int(model_type)](weights='imagenet')
    if model_type in SMALL_SCORE_MODELS:
        score_range_type = SCORE_RANGES_SMALL
    elif model_type in BIG_SCORE_MODELS:
        score_range_type = SCORE_RANGES_BIG
    else:
        exit("Error 3")

# Define input directory
if input_folder is None:
    input_folder = get_folder_path("Choose your input folder")

# Define output directory
if output_folder is None:
    output_folder = get_folder_path("Choose your output folder")

if mode != 16:
    if move_or_copy is None:
        while True:
            move_or_copy = input("Do you wanna move or copy the files?\n1 = Move\n2 = Copy\nSelected mode: ")

            if move_or_copy == "1" or move_or_copy == "2":
                move_or_copy = int(move_or_copy)
                break

            invalid_input()

if autonomous != "True":
    while True:
        save_config = input("Do you wanna save current settings for next time? (y = yes, n = no) ")
        if save_config == "y" or save_config == "n":
            break

        invalid_input()

    if save_config == "y":
        data = {
            "autonomous": autonomous,
            "input_folder": input_folder,
            "output_folder": output_folder,
            "move_or_copy": move_or_copy,
            "mode": mode,
            "experimental": experimental,
            "own_parameters": own_parameters,
            "parameters": [
                parameters
            ],
            "strict_parameters": strict_parameters,
            "split_words": split_words,
            "model_type": model_type,
            "score_or_class": score_or_class
        }

        with open('config.json', 'w') as file:
            json.dump(data, file, indent=4)

# Count the total number of images in the input folder
valid_extensions = ('.png', '.jpg', '.jpeg')
image_files = [file_path for file_path in input_folder.rglob('*') if file_path.suffix.lower() in valid_extensions]
total_images = len(image_files)

for idx, file_path in enumerate(image_files):
    # Check if the file is a valid image
    if file_path.suffix.lower() in valid_extensions:
        print(f"\nAnalyzing image {idx + 1}/{total_images}\n{file_path.name}")
        if mode in [1, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15]:
            # Check if the image is NSFW
            nsfw_probability = is_nsfw(file_path)
            print(f"NSFW probability: {nsfw_probability}")
            nsfw_folder_name = get_folder_name(nsfw_probability, NSFW_RANGES)
            nsfw_folder_name = "X" + nsfw_folder_name

        if mode in [2, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15]:
            # Get the score and index for the input image
            class_index, score = get_image_score(file_path)
            print(f"Score: {score}, Class: {class_index}")
            if score_or_class == "s":
                score_folder_name = get_folder_name(score, score_range_type)
                score_folder_name = "S" + score_folder_name
            elif score_or_class == "c":
                score_folder_name = "C" + class_index

        if mode in [3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            # Extract the model name
            model_name = extract_model_name(file_path)
            print(f"Model: {model_name}")
            model_folder_name = model_name

        if mode in [16]:
            parameter_list = extract_parameters(file_path)
            print(f"Parameters: {parameter_list}")

        mode_folders = {
            1: [nsfw_folder_name],
            2: [score_folder_name],
            3: [model_folder_name],
            4: [nsfw_folder_name, score_folder_name],
            5: [nsfw_folder_name, model_folder_name],
            6: [score_folder_name, nsfw_folder_name],
            7: [score_folder_name, model_folder_name],
            8: [model_folder_name, nsfw_folder_name],
            9: [model_folder_name, score_folder_name],
            10: [nsfw_folder_name, score_folder_name, model_folder_name],
            11: [nsfw_folder_name, model_folder_name, score_folder_name],
            12: [score_folder_name, nsfw_folder_name, model_folder_name],
            13: [score_folder_name, model_folder_name, nsfw_folder_name],
            14: [model_folder_name, nsfw_folder_name, score_folder_name],
            15: [model_folder_name, score_folder_name, nsfw_folder_name],
        }

        if mode in mode_folders:
            new_output_folder = output_folder
            for folder_name in mode_folders[mode]:
                new_output_folder = new_output_folder / folder_name
                new_output_folder.mkdir(parents=True, exist_ok=True)
            destination_file_path = new_output_folder / file_path.name
            if not destination_file_path.exists():
                if move_or_copy == 1:
                    print(f"Image: {file_path.name} -> Move to folder: {new_output_folder}")
                    shutil.move(file_path, destination_file_path)
                    print(f"Moved image to {destination_file_path}")
                elif move_or_copy == 2:
                    print(f"Image: {file_path.name} -> Copy to folder: {new_output_folder}")
                    shutil.copy(file_path, destination_file_path)
                    print(f"Copied image to {destination_file_path}")
                else:
                    exit("Error 4")
            else:
                print(f"Skipped image '{file_path.name}' as it already exists in the destination folder.")

        elif mode == 16:
            if strict_parameters:
                parameters_found = True
                for parameter in parameters:
                    if parameter not in parameter_list:
                        parameters_found = False
                        break

                if parameters_found:
                    folder_name = "_".join(parameters)  # Concatenate parameters with underscores
                    new_output_folder = output_folder / folder_name
                    new_output_folder.mkdir(parents=True, exist_ok=True)
                    destination_file_path = new_output_folder / file_path.name
                    if not destination_file_path.exists():
                        print(f"Image: {file_path.name} -> Move to folder: {new_output_folder}")
                        shutil.copy(file_path, destination_file_path)
                        print(f"Copied image to {destination_file_path}")
                elif not parameters_found:
                    print("No Matching parameter(s) found. Skipping image")
                else:
                    exit("Error 8")

            else:
                for parameter in parameter_list:
                    if parameters is None or parameter in parameters:
                        new_output_folder = output_folder / parameter
                        new_output_folder.mkdir(parents=True, exist_ok=True)
                        destination_file_path = new_output_folder / file_path.name
                        if not destination_file_path.exists():
                            print(f"Image: {file_path.name} -> Move to folder: {new_output_folder}")
                            shutil.copy(file_path, destination_file_path)
                            print(f"Copied image to {destination_file_path}")

        else:
            print("Invalid mode entered.")
    else:
        print(f"Skipping non-image file: {file_path.name}")

print("Image analysis and sorting complete.")
exit()
