import os
import sqlite3
from PIL import Image
import re


# Function to extract specific metadata subcategories from the "Parameters" field
def extract_parameters_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            metadata = img.info.get('Parameters', '')

            # Initialize a dictionary to store the subcategories
            metadata_dict = {}

            # Define the subcategories to extract
            subcategories = [
                "Negative prompt", "Steps", "Sampler", "CFG Scale", "Seed",
                "Size", "Model hash", "Model", "Seed resize from", "Denoising strength", "Version"
            ]

            # Extract specific metadata subcategories
            for subcategory in subcategories:
                match = re.search(rf'{subcategory}:\s*([^,]+)', metadata)
                if match:
                    metadata_dict[subcategory] = match.group(1).strip()

            return metadata_dict
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return None


# Function to extract text between "Parameters" and "Negative prompt"
def extract_text_between_parameters_and_negative_prompt(metadata):
    # Use regular expressions to extract the text between "Parameters" and "Negative prompt"
    match = re.search(r'Parameters:(.*?)(?:Negative prompt|$)', metadata, re.DOTALL)
    if match:
        parameters_text = match.group(1).strip()
        return parameters_text
    else:
        return None


# Function to check if the database exists
def database_exists(database_name):
    return os.path.exists(database_name)


# Function to create a SQLite database and table
def create_database(database_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ImageMetadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            PositivePrompt TEXT,
            NegativePrompt TEXT,
            Steps TEXT,
            Sampler TEXT,
            CFGScale TEXT,
            Seed TEXT,
            Size TEXT,
            ModelHash TEXT,
            Model TEXT,
            SeedResizeFrom TEXT,
            DenoisingStrength TEXT,
            Version TEXT
        )
    ''')
    conn.commit()
    return conn


# Function to insert metadata into the database
def insert_metadata_into_database(conn, filename, metadata):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO ImageMetadata (
            filename, PositivePrompt, NegativePrompt, Steps, Sampler, CFGScale, Seed, Size, ModelHash,
            Model, SeedResizeFrom, DenoisingStrength, Version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename,
        metadata.get('Positive prompt', ''),
        metadata.get('Negative prompt', ''),
        metadata.get('Steps', ''),
        metadata.get('Sampler', ''),
        metadata.get('CFG Scale', ''),
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

# SQLite database file
database_name = 'image_metadata.db'

# Create a database and table if it doesn't exist
if not database_exists(database_name):
    conn = create_database(database_name)
else:
    conn = sqlite3.connect(database_name)

# Loop through the images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        metadata = extract_parameters_metadata(image_path)

        if metadata is not None:
            positive_prompt_text = extract_text_between_parameters_and_negative_prompt(metadata.get('Parameters', ''))
            metadata['Positive prompt'] = positive_prompt_text
            insert_metadata_into_database(conn, filename, metadata)
            print(f"Metadata from {filename} extracted and added to the database.")

# Close the database connection
conn.close()
