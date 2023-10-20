import os
from PIL import Image
import mysql.connector


# Function to extract metadata categories and subcategories
def get_image_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            metadata = img.info
        return metadata
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {}


# Split by the first occurrence of "Negative prompt" or "Steps"
    negative_prompt_index = metadata_str.find("Negative prompt:")
    steps_index = metadata_str.find("Steps:")

    if negative_prompt_index != -1:
        positive_prompt = metadata_str[:negative_prompt_index].strip()
        metadata_dict["Positive Prompt"] = positive_prompt

        remaining_content = metadata_str[negative_prompt_index:].strip()
        negative_prompt_end = remaining_content.find("Steps:")

        if negative_prompt_end != -1:
            negative_prompt = remaining_content[:negative_prompt_end].strip()
            metadata_dict["Negative Prompt"] = negative_prompt
        else:
            metadata_dict["Negative Prompt"] = remaining_content

        steps_section = metadata_str[steps_index:].strip()
        metadata_dict["Steps"] = steps_section

        # Split the content after "Steps:" into key-value pairs
        content_segments = steps_section.split(", ")
        for segment in content_segments:
            key_value = segment.split(": ", 1)
            if len(key_value) == 2:
                key, value = key_value[0], key_value[1]
                metadata_dict[key] = value
    elif steps_index != -1:
        positive_prompt = metadata_str[:steps_index].strip()
        metadata_dict["Positive Prompt"] = positive_prompt

        steps_section = metadata_str[steps_index:].strip()
        metadata_dict["Steps"] = steps_section

        # Split the content after "Steps:" into key-value pairs
        content_segments = steps_section.split(", ")
        for segment in content_segments:
            key_value = segment.split(": ", 1)
            if len(key_value) == 2:
                key, value = key_value[0], key_value[1]
                metadata_dict[key] = value
    else:
        # If neither "Negative prompt" nor "Steps" is found, consider the entire section as "Positive prompt"
        metadata_dict["Positive Prompt"] = metadata_str

    return metadata_dict


# Function to create a MySQL database and table
def connect_database(database_name):
    conn = mysql.connector.connect(
        host="your_host",
        user="your_username",
        password="your_password",
        database=database_name
    )
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ImageMetadata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            FileName VARCHAR(255),
            Directory TEXT,
            FileSize TEXT,
            PositivePrompt TEXT,
            NegativePrompt TEXT,
            Steps TEXT,
            Sampler TEXT,
            CFGScale TEXT,
            Seed TEXT,
            ImageSize TEXT,
            ModelHash TEXT,
            Model TEXT,
            SeedResizeFrom TEXT,
            DenoisingStrength TEXT,
            Version TEXT
        )
    ''')
    conn.commit()
    return conn


# Function to insert metadata into the MySQL database
def insert_metadata_into_database(conn, metadata):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO ImageMetadata (
            FileName, Directory, FileSize, PositivePrompt, NegativePrompt, Steps, Sampler, CFGScale, Seed, ImageSize,
            ModelHash, Model, SeedResizeFrom, DenoisingStrength, Version
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        metadata.get('File Name', ''),
        metadata.get('Directory', ''),
        metadata.get('File Size', ''),
        metadata.get('Positive prompt', ''),
        metadata.get('Negative prompt', ''),
        metadata.get('Steps', ''),
        metadata.get('Sampler', ''),
        metadata.get('CFG scale', ''),
        metadata.get('Seed', ''),
        metadata.get('Size', ''),
        metadata.get('Model hash', ''),
        metadata.get('Model', ''),
        metadata.get('Seed resize from', ''),
        metadata.get('Denoising strength', ''),
        metadata.get('Version', '')
    ))
    conn.commit()


# Folder containing images
image_folder = 'path/to/your/image/folder'

# MySQL database configuration
database_name = 'image_metadata'

# Create a MySQL database and table if it doesn't exist
conn = connect_database(database_name)

# Loop through the images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        metadata = get_image_metadata(image_path)
        parameters_metadata = metadata.get("parameters", "")
        extracted_metadata = extract_metadata_from_parameter(parameters_metadata, image_path)

        print("Extracted Metadata:")
        for key, value in extracted_metadata.items():
            print(f"{key}: {value}")

        if extracted_metadata is not None:
            insert_metadata_into_database(conn, extracted_metadata)
            print(f"Metadata from {filename} extracted and added to the database.")

# Close the database connection
conn.close()
