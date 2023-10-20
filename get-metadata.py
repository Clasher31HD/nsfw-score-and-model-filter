from PIL import Image
import os


def get_png_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            info = img.info
            print("PNG Metadata:")
            for key, value in info.items():
                print(f"{key}: {value}")

        file_name = os.path.basename(image_path)
        directory = os.path.dirname(image_path)
        file_size = os.path.getsize(image_path)

        print("\nAdditional File Information:")
        print(f"File Name: {file_name}")
        print(f"Directory: {directory}")
        print(f"File Size: {file_size} bytes")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with the path to your image
    get_png_metadata(image_path)
