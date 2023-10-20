from PIL import Image


def extract_metadata_from_parameter(metadata_str):
    metadata_dict = {"Positive prompt": ""}
    metadata_str = metadata_str.split('.')

    for segment in metadata_str:
        key_value = segment.split(': ')
        if len(key_value) == 2:
            key, value = key_value[0], key_value[1]
            if key == "Positive prompt":
                metadata_dict["Positive prompt"] = value
            else:
                metadata_dict[key] = value

    return metadata_dict


if __name__ == "__main__":
    metadata_str = "Hallo, ich, bin cool.Negative prompt: Ugly.Steps: 50.Sampler: Euler"
    metadata = extract_metadata_from_parameter(metadata_str)

    print("Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
