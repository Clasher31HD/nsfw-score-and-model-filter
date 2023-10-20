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


def extract_metadata_from_parameter(metadata_str, image_path):
    metadata_dict = {}

    # Add filename, directory, and file size to the metadata
    file_name = os.path.basename(image_path)
    directory = os.path.dirname(image_path)
    file_size = os.path.getsize(image_path)
    metadata_dict["File Name"] = file_name.strip(".png")
    metadata_dict["Directory"] = directory.rsplit('/', 2)[-2]
    metadata_dict["File Size"] = file_size

    # Try to split by "Negative prompt" first
    sections = metadata_str.split("Negative prompt:")

    if len(sections) == 2:
        positive_prompt = sections[0].strip()
        negative_and_steps = sections[1].strip().split("Steps:", 1)

        metadata_dict["Positive Prompt"] = positive_prompt
        negative_prompt = negative_and_steps[0].strip()

        # Exclude the label "Negative prompt:" from the content
        if negative_prompt.startswith("Negative prompt:"):
            negative_prompt = negative_prompt[len("Negative prompt:"):]

        metadata_dict["Negative Prompt"] = negative_prompt

        if len(negative_and_steps) == 2:
            steps_and_content = "Steps:" + negative_and_steps[1].strip()

            metadata_dict["Steps"] = steps_and_content.strip()

            # Split the content after "Steps:" into key-value pairs
            content_segments = steps_and_content.split(", ")
            for segment in content_segments:
                key_value = segment.split(": ", 1)
                if len(key_value) == 2:
                    key, value = key_value[0], key_value[1]
                    metadata_dict[key] = value
    else:
        # If "Negative prompt" is not found, try splitting by "Steps"
        sections = metadata_str.split("Steps:", 1)

        if len(sections) == 2:
            positive_prompt = sections[0].strip()
            steps_and_content = "Steps:" + sections[1].strip()

            metadata_dict["Positive Prompt"] = positive_prompt
            metadata_dict["Steps"] = steps_and_content.strip()

            # Split the content after "Steps:" into key-value pairs
            content_segments = steps_and_content.split(", ")
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
