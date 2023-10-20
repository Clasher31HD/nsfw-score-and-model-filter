from PIL import Image
import os


def get_image_metadata(image_path):
    try:
        metadata_dict = {
            "parameter": {},
            "File Name": "",
            "Directory": "",
            "File Size": 0
        }

        with Image.open(image_path) as img:
            info = img.info
            for key, value in info.items():
                if key in ["Negative prompt", "Steps", "Sampler", "CFG Scale", "Seed", "Model hash", "Model",
                           "Seed resize from", "Denoising strength", "Version"]:
                    metadata_dict["Positive prompt"][key] = value
                else:
                    metadata_dict[key] = value

        file_name = os.path.basename(image_path)
        directory = os.path.dirname(image_path)
        file_size = os.path.getsize(image_path)

        metadata_dict["File Name"] = file_name
        metadata_dict["Directory"] = directory
        metadata_dict["File Size"] = file_size

        return metadata_dict
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {}


if __name__ == "__main__":
    image_path = "path_to_your_image.png"  # Replace with the path to your PNG image
    all_metadata = get_image_metadata(image_path)

    if all_metadata:
        print("All Metadata Values:")
        for key, value in all_metadata.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
