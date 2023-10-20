from PIL import Image
import os


def get_image_metadata(image_path):
    try:
        metadata_list = []

        with Image.open(image_path) as img:
            info = img.info
            for key, value in info.items():
                metadata_list.append(f"{key}: {value}")

        file_name = os.path.basename(image_path)
        directory = os.path.dirname(image_path)
        file_size = os.path.getsize(image_path)

        additional_info = [
            f"File Name: {file_name}",
            f"Directory: {directory}",
            f"File Size: {file_size} bytes"
        ]

        metadata_list.extend(additional_info)

        return metadata_list
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []


if __name__ == "__main__":
    image_path = "path_to_your_image.png"  # Replace with the path to your PNG image
    all_metadata = get_image_metadata(image_path)

    if all_metadata:
        print("All Metadata Values:")
        for item in all_metadata:
            print(item)
