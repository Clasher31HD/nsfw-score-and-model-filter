import os
from PIL import Image
import re
import mysql.connector


# Function to extract metadata categories and subcategories
def extract_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            metadata_dict = {}

            # Extract "File Name," "File Size," and "Directory"
            file_name = img.info.get('File Name', '')
            if file_name:
                match = re.search(r'File Name:\s*([^,]+)', file_name)
                if match:
                    metadata_dict['File Name'] = match.group(1).strip()

            directory = img.info.get('Directory', '')
            if directory:
                match = re.search(r'Directory:\s*([^,]+)', directory)
                if match:
                    metadata_dict['Directory'] = match.group(1).strip()

            file_size = img.info.get('File Size', '')
            if file_size:
                match = re.search(r'File Size:\s*([^,]+)', file_size)
                if match:
                    metadata_dict['File Size'] = match.group(1).strip()

            image_size = img.info.get('Image Size', '')
            if image_size:
                match = re.search(r'Image Size:\s*([^,]+)', image_size)
                if match:
                    metadata_dict['Image Size'] = match.group(1).strip()

            # Extract the beginning of the "Parameters" category
            parameters = img.info.get('Parameters', '')
            if parameters:
                match = re.search(r'Parameters:(.*?)(?:Negative prompt|$)', parameters, re.DOTALL)
                if match:
                    parameters_text = match.group(1).strip()
                    metadata_dict['Parameters'] = parameters_text

            # Define the subcategories to extract from "Parameters"
            subcategories = [
                "Negative prompt", "Steps", "Sampler", "CFG Scale", "Seed",
                "Model hash", "Model", "Seed resize from", "Denoising strength", "Version"
            ]

            # Extract specific metadata subcategories from "Parameters"
            for subcategory in subcategories:
                match = re.search(rf'{subcategory}:\s*([^,]+)', parameters)
                if match:
                    metadata_dict[subcategory] = match.group(1).strip()

            return metadata_dict
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return None


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
            Image Size TEXT,
            PositivePrompt TEXT,
            NegativePrompt TEXT,
            Steps TEXT,
            Sampler TEXT,
            CFGScale TEXT,
            Seed TEXT,
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
            filename, PositivePrompt, NegativePrompt, Steps, Sampler, CFGScale, Seed, Size, ModelHash,
            Model, SeedResizeFrom, DenoisingStrength, Version
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        metadata.get('File Name', ''),
        metadata.get('Directory', ''),
        metadata.get('File Size', ''),
        metadata.get('Image Size', ''),
        metadata.get('Positive prompt', ''),
        metadata.get('Negative prompt', ''),
        metadata.get('Steps', ''),
        metadata.get('Sampler', ''),
        metadata.get('CFG Scale', ''),
        metadata.get('Seed', ''),
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
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        metadata = extract_metadata(image_path)

        if metadata is not None:
            insert_metadata_into_database(conn, metadata)
            print(f"Metadata from {filename} extracted and added to the database.")

# Close the database connection
conn.close()
