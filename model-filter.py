from PIL import Image
import os
import shutil


def extract_model_name(metadata):
    # Extract the model name from the metadata dictionary
    params = metadata.get("parameters", "")
    model_start = params.find("Model:") + len("Model:")
    model_end = params.find(",", model_start)
    model_name = params[model_start:model_end].strip()
    return model_name


def filter_images_by_model(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            file_path = os.path.join(input_directory, filename)
            try:
                image = Image.open(file_path)
                metadata = image.info
                model_name = extract_model_name(metadata)
                if model_name:
                    model_folder = os.path.join(output_directory, model_name)
                    os.makedirs(model_folder, exist_ok=True)
                    new_file_path = os.path.join(model_folder, filename)
                    shutil.copyfile(file_path, new_file_path)
                    print("Image:", filename)
                    print("Model:", model_name)
                    print("Copied to:", new_file_path)
                    print("------------------------")
                else:
                    print("Model name not found in metadata for:", filename)
            except Exception as e:
                print("Error processing image:", filename)
                print("Error message:", str(e))


def process_images_folder(input_folder_path, output_folder_path):
    if not os.path.exists(input_folder_path) or not os.path.isdir(input_folder_path):
        print("Invalid input folder path:", input_folder_path)
        return
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print("Output folder created:", output_folder_path)
    elif not os.path.isdir(output_folder_path):
        print("Invalid output folder path:", output_folder_path)
        return
    filter_images_by_model(input_folder_path, output_folder_path)


# Example usage
input_folder_path = "E:/Stable-Diffusion/stable-diffusion-webui/horde/2023-06-21"
output_folder_path = "E:/Filtered-Images"
process_images_folder(input_folder_path, output_folder_path)
