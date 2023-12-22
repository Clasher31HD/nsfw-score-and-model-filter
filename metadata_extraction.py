import os
import logging
from PIL import Image
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import mysql.connector
import yaml


def read_configuration():
    try:
        with open("metadata_config.yml", "r") as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError("metadata_config.yml file not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML in yts_config.yml: {e}")
    

def setup_logger():
    # General Settings
    config = read_configuration()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    level = config["level"]
    logs_directory = config["logs_directory"]

    # Standard Logger
    logger_log_file = os.path.join(logs_directory, "Info.log")
    logger_file_handler = logging.FileHandler(logger_log_file)
    logger_file_handler.setFormatter(formatter)
    logger_file_handler.setLevel(level)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.addHandler(logger_file_handler)

    # Extraction Logger
    extraction_log_file = os.path.join(logs_directory, "Extraction.log")
    extraction_file_handler = logging.FileHandler(extraction_log_file)
    extraction_file_handler.setFormatter(formatter)
    extraction_file_handler.setLevel(level)
    extraction_logger = logging.getLogger('extraction')
    extraction_logger.setLevel(level)
    extraction_logger.addHandler(extraction_file_handler)

    # NSFW Logger 
    nsfw_log_file = os.path.join(logs_directory, "NSFW.log")
    nsfw_file_handler = logging.FileHandler(nsfw_log_file)
    nsfw_file_handler.setFormatter(formatter)
    nsfw_file_handler.setLevel(level)
    nsfw_logger = logging.getLogger('nsfw')
    nsfw_logger.setLevel(level)
    nsfw_logger.addHandler(nsfw_file_handler)

    return logger, extraction_logger, nsfw_logger


# Function to extract metadata categories and subcategories
def get_image_metadata(image_path, logger):
    try:
        with Image.open(image_path) as img:
            metadata = img.info
        return metadata
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {}


def extract_metadata_from_parameter(metadata_str, image_path, nsfw, logger, nsfw_logger):
    metadata_dict = {}

    if nsfw:
        import opennsfw2 as n2
        try:
            img = Image.open(image_path)
            img.thumbnail((512, 512))

            nsfw_probability = n2.predict_image(image_path)
            nsfw_logger.info(f"NSFWProbability is {nsfw_probability}")

            metadata_dict["NSFWProbability"] = nsfw_probability
        except OSError as e:
            nsfw_logger.warning(f"Skipping image '{image_path.name}' due to an error: {str(e)}")
    else:
        nsfw_probability = "Unknown"
        metadata_dict["NSFWProbability"] = nsfw_probability
        nsfw_logger.info("NSFW is off so no nsfw calculation")

    hashermd5 = hashlib.md5()
    hashersha1 = hashlib.sha1()
    hashersha256 = hashlib.sha256()

    # Get Hash values
    with open(image_path, "rb") as file:
        while True:
            chunk = file.read(4096)  # Read in 4KB chunks
            if not chunk:
                break
            hashermd5.update(chunk)
            hashersha1.update(chunk)
            hashersha256.update(chunk)

    hash_md5 = hashermd5.hexdigest()
    hash_sha1 = hashersha1.hexdigest()
    hash_sha256 = hashersha256.hexdigest()
    metadata_dict["MD5"] = hash_md5
    metadata_dict["SHA1"] = hash_sha1
    metadata_dict["SHA256"] = hash_sha256

    # Add filename, directory, and file size to the metadata
    file_name = os.path.basename(image_path).strip(".png")
    directory = os.path.basename(os.path.dirname(image_path))
    file_size = os.path.getsize(image_path)
    metadata_dict["File Name"] = file_name
    metadata_dict["Directory"] = directory
    metadata_dict["File Size"] = file_size

    # Split by the first occurrence of "Negative prompt" or "Steps"
    negative_prompt_index = metadata_str.find("Negative prompt:")
    steps_index = metadata_str.find("Steps:")

    if negative_prompt_index != -1:
        positive_prompt = metadata_str[:negative_prompt_index].strip()
        metadata_dict["Positive prompt"] = positive_prompt

        remaining_content = metadata_str[negative_prompt_index:].strip()
        negative_prompt_end = remaining_content.find("Steps:")

        if negative_prompt_end != -1:
            negative_prompt = remaining_content[len("Negative prompt:"):negative_prompt_end].strip()
            metadata_dict["Negative prompt"] = negative_prompt
        else:
            metadata_dict["Negative prompt"] = remaining_content

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
        metadata_dict["Positive prompt"] = positive_prompt

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
        metadata_dict["Positive prompt"] = metadata_str

    return metadata_dict


# Function to create a MySQL database and table
def connect_database(host, user, password, database_name, table_name):
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database_name
    )
    cursor = conn.cursor()
    create_table_query = f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
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
            Version TEXT,
            NSFWProbability TEXT,
            MD5 TEXT,
            SHA1 TEXT,
            SHA256 TEXT
        )
    '''
    cursor.execute(create_table_query)
    conn.commit()
    return conn


# Function to insert metadata into the MySQL database if it doesn't already exist
def insert_metadata_into_database(conn, table, metadata, logger):
    cursor = conn.cursor()

    # Check if the combination of FileName and Directory already exists in the database
    query = f'''
    SELECT COUNT(*) FROM {table}
    WHERE SHA256 = %s
    '''
    cursor.execute(query, (metadata.get('SHA256', ''),))
    result = cursor.fetchone()

    if result[0] == 0:
        # The combination doesn't exist, so insert the metadata
        cursor.execute(f'''
            INSERT INTO {table} (
                FileName, Directory, FileSize, PositivePrompt, NegativePrompt, Steps, Sampler, CFGScale, Seed, 
                ImageSize, ModelHash, Model, SeedResizeFrom, DenoisingStrength, Version, NSFWProbability, MD5, SHA1, SHA256
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            metadata.get('Version', ''),
            metadata.get('NSFWProbability', ''),
            metadata.get('MD5', ''),
            metadata.get('SHA1', ''),
            metadata.get('SHA256', '')
        ))
        conn.commit()
        logger.info(f"Metadata from {metadata.get('File Name', '')} extracted and added to the database.")
    else:
        # The combination already exists, so skip the insert
        logger.warning(f"Metadata from {metadata.get('File Name', '')} already exists in the database. Skipping insert.")


def start_metadata_extractor():
    logger, extraction_logger, nsfw_logger = setup_logger()
    try:
        try:
            logger.info("Script started.")
            
            config = read_configuration()
            host = config["host"]
            user = config["user"]
            password = config["password"]
            database_name = config["database_name"]
            table_name = config["table_name"]
            image_folder = Path(config["image_folder"])
            use_yesterday = config.get("use_yesterday", False)
            nsfw = config.get("nsfw_probability", True)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid configuration: {str(e)}")

        if use_yesterday == "True":
            today = datetime.now()
            yesterday = today - timedelta(days=1)
            formatted_yesterday = yesterday.strftime("%Y-%m-%d")
            image_folder = os.path.join(image_folder, formatted_yesterday)

        # Create a MySQL database and table if it doesn't exist
        conn = connect_database(host, user, password, database_name, table_name)

        # Loop through the images in the folder
        for root, dirs, files in os.walk(image_folder):
            for filename in files:
                if filename.endswith('.png'):
                    image_path = os.path.join(root, filename)
                    metadata = get_image_metadata(image_path, logger)
                    parameters_metadata = metadata.get("parameters", "")
                    extracted_metadata = extract_metadata_from_parameter(parameters_metadata, image_path, nsfw, logger, nsfw_logger)

                    if extracted_metadata is not None:
                        insert_metadata_into_database(conn, table_name, extracted_metadata, logger)

        # Close the database connection
        conn.close()
        logger.info("Script finished.")
    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))

if __name__ == "__main__":
    start_metadata_extractor()
