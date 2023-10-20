from PIL import Image
import os


def get_image_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            metadata = img.info
        return metadata
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {}


def extract_metadata_from_parameter(metadata_str, image_path):
    metadata_dict = {}

    # Add filename, directory, and file size to the metadata
    file_name = os.path.basename(image_path)
    directory = os.path.dirname(image_path)
    file_size = os.path.getsize(image_path)
    metadata_dict["File Name"] = file_name
    metadata_dict["Directory"] = directory
    metadata_dict["File Size"] = f"{file_size} bytes"

    # Split by the first occurrence of "Steps:"
    sections = metadata_str.split("Steps: ", 1)

    if len(sections) == 2:
        positive_prompt = sections[0].strip()
        steps_and_content = sections[1].strip()

        # Include the "Steps" label and its content
        metadata_dict["Positive prompt"] = positive_prompt
        metadata_dict["Steps"] = steps_and_content.split(", ")[0]

        # Split the content after "Steps:" into key-value pairs
        content_segments = steps_and_content.split(", ")
        for segment in content_segments:
            key_value = segment.split(": ", 1)
            if len(key_value) == 2:
                key, value = key_value[0], key_value[1]
                metadata_dict[key] = value

    return metadata_dict


if __name__ == "__main__":
    image_path = "path_to_your_image.png"  # Replace with the path to your PNG image
    metadata = get_image_metadata(image_path)

    parameters_metadata = metadata.get("parameters", "")
    extracted_metadata = extract_metadata_from_parameter(parameters_metadata, image_path)

    print("Extracted Metadata:")
    for key, value in extracted_metadata.items():
        print(f"{key}: {value}")
