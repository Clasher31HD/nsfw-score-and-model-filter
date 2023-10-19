from PIL import Image


def get_png_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            info = img.info
            print("PNG Metadata:")
            for key, value in info.items():
                print(f"{key}: {value}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with the path to your image
    get_png_metadata(image_path)
