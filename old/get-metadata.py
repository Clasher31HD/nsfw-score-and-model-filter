from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def get_image_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            metadata = img._getexif()
            if metadata:
                print("Image Metadata:")
                for tag, value in metadata.items():
                    tag_name = TAGS.get(tag, tag)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except UnicodeDecodeError:
                            pass
                    print(f"{tag_name}: {value}")
            else:
                print("No metadata found in the image.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with the path to your image
    get_image_metadata(image_path)
