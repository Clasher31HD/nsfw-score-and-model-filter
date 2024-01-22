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
    log_by_day = config.get("log_by_day", True)
    if log_by_day:
        year = str(datetime.now().strftime('%Y'))
        month = str(datetime.now().strftime('%m'))
        day = str(datetime.now().strftime('%d'))
        directory_path = os.path.join(logs_directory, year, month, day)
        os.makedirs(directory_path, exist_ok=True)
        logger_log_file = os.path.join(directory_path, f"{year}-{month}-{day}-Info.log")
        extraction_log_file = os.path.join(directory_path, f"{year}-{month}-{day}-Extraction.log")
        nsfw_log_file = os.path.join(directory_path, f"{year}-{month}-{day}-NSFW.log")
    else:
        logger_log_file = os.path.join(logs_directory, "Info.log")
        extraction_log_file = os.path.join(logs_directory, "Extraction.log")
        nsfw_log_file = os.path.join(logs_directory, "NSFW.log")

    # Standard Logger
    logger_file_handler = logging.FileHandler(logger_log_file)
    logger_file_handler.setFormatter(formatter)
    logger_file_handler.setLevel(level)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.addHandler(logger_file_handler)

    # Extraction Logger
    extraction_file_handler = logging.FileHandler(extraction_log_file)
    extraction_file_handler.setFormatter(formatter)
    extraction_file_handler.setLevel(level)
    extraction_logger = logging.getLogger('extraction')
    extraction_logger.setLevel(level)
    extraction_logger.addHandler(extraction_file_handler)

    # NSFW Logger 
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
            if metadata:
                if metadata is not None:
                    return metadata
                else:
                    logger.error(f"Metadata from file {image_path} is not available")
                    return {}
            else:
                logger.error(f"Metadata from file {image_path} is not available")
                return {}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {}


def extract_metadata_from_parameter(metadata_str, image_path, nsfw, logger, nsfw_logger):
    if metadata_str is None:
        logger.error("metadata_str is None.")
        return {}
    
    metadata_dict = {}

    if nsfw:
        import opennsfw2 as n2
        try:
            img = Image.open(image_path)
            img.thumbnail((512, 512))

            nsfw_probability = n2.predict_image(image_path)
            if nsfw_probability is None:
                logger.error(f"An error occurred: {str(e)}")
                return {}
            
            nsfw_logger.info(f"NSFWProbability for image '{os.path.basename(image_path)}' is {nsfw_probability}")
            metadata_dict["NSFWProbability"] = nsfw_probability
        except OSError as e:
            nsfw_logger.warning(f"Skipping image '{os.path.basename(image_path)}' due to an error: {str(e)}")
    else:
        nsfw_probability = ""
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
    creation_time = datetime.fromtimestamp(os.path.getctime(image_path))
    metadata_dict["File Name"] = file_name
    metadata_dict["Directory"] = directory
    metadata_dict["File Size"] = file_size
    metadata_dict["Created At"] = creation_time

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
def connect_database(host, user, password, database_name, table_name, logger):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database_name
        )
        cursor = conn.cursor()
    except mysql.connector.Error as e:
        logger.error(f"Failed to connect to the database: {e}")
        return None, []

    # Check if the table exists
    cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
    table_exists = cursor.fetchone()

    if not table_exists:
        # Create the table if it doesn't exist
        create_table_query = f'''
            CREATE TABLE {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                FileName VARCHAR(255),
                Directory TEXT,
                FileSize TEXT,
                CreatedAt TEXT,
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

        try:
            cursor.execute(create_table_query)
            conn.commit()
        except mysql.connector.Error as e:
            logger.error(f"Table creation could not be executed: {e}")
            return None, []
    
        logger.info(f"Table {table_name} created successfully.")
    else:
        # Check and add columns if they do not exist
        expected_columns = [
            'FileName', 'Directory', 'FileSize', 'CreatedAt', 'PositivePrompt', 'NegativePrompt', 'Steps',
            'Sampler', 'CFGScale', 'Seed', 'ImageSize', 'ModelHash',
            'Model', 'SeedResizeFrom', 'DenoisingStrength', 'Version',
            'NSFWProbability', 'MD5', 'SHA1', 'SHA256'
        ]

        cursor.execute(f"DESCRIBE {table_name}")
        existing_columns = [column[0] for column in cursor.fetchall()]

        for column in expected_columns:
            if column not in existing_columns:
                # Add the column if it does not exist
                add_column_query = f"ALTER TABLE {table_name} ADD COLUMN {column} TEXT"
                try:
                    cursor.execute(add_column_query)
                    conn.commit()
                    logger.info(f"Column {column} added successfully.")
                except mysql.connector.Error as e:
                    logger.error(f"Error adding column {column}: {e}")

    return conn, existing_columns


# Function to insert metadata into the MySQL database if it doesn't already exist
def insert_metadata_into_database(conn, table, existing_columns, metadata, logger, extraction_logger):
    cursor = conn.cursor()

    if metadata is None:
        logger.error(f"No metadata found for {table}.")
        return 
    
    # Check if the data already exists in the database
    query = f'''
    SELECT * FROM {table}
    WHERE SHA256 = %s
    '''
    cursor.execute(query, (metadata.get('SHA256', ''),))
    existing_record = list(cursor.fetchone())

    if existing_record is None:
        logger.error(f"No existing record found for {table}.")
        return

    if existing_record:
        # Iterate through metadata fields and update the database record if necessary
        for field_name, value in metadata.items():
            # Check if the field is in the existing columns
            if field_name in existing_columns:
                # Get the index of the field
                field_index = existing_columns.index(field_name)
                
                # Check if the values are different
                if existing_record[field_index] != value:
                    # Update the value in the existing record
                    existing_record[field_index] = value

        # Update the database record
        update_query = f'''
        UPDATE {table}
        SET
            FileName = %s,
            Directory = %s,
            FileSize = %s,
            CreatedAt = %s,
            PositivePrompt = %s,
            NegativePrompt = %s,
            Steps = %s,
            Sampler = %s,
            CFGScale = %s,
            Seed = %s,
            ImageSize = %s,
            ModelHash = %s,
            Model = %s,
            SeedResizeFrom = %s,
            DenoisingStrength = %s,
            Version = %s,
            NSFWProbability = %s,
            MD5 = %s,
            SHA1 = %s,
            SHA256 = %s
        WHERE SHA256 = %s
        '''
        try:
            cursor.execute(update_query, tuple(existing_record + [existing_record[existing_columns.index('SHA256')]]))
            conn.commit()
        except:
            extraction_logger.error(f"Failed to update metadata for {metadata.get('File Name', '')} in folder {metadata.get('Directory', '')} in the database.")
            extraction_logger.error(f"{cursor.status}")
            extraction_logger.error(f"{cursor.description}")
            extraction_logger.error(f"{cursor.diagnostics}")

        extraction_logger.info(f"Metadata for {metadata.get('File Name', '')} in folder {metadata.get('Directory', '')} has been updated in the database.")
    else:
        # The combination doesn't exist, so insert the metadata
        try:
            cursor.execute(f'''
            INSERT INTO {table} (
                FileName, Directory, FileSize, CreatedAt, PositivePrompt, NegativePrompt, Steps, Sampler, CFGScale, Seed, 
                ImageSize, ModelHash, Model, SeedResizeFrom, DenoisingStrength, Version, NSFWProbability, MD5, SHA1, SHA256
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                metadata.get('File Name', ''),
                metadata.get('Directory', ''),
                metadata.get('File Size', ''),
                metadata.get('Created at', ''),
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
        except:
            logger.error(f"Error while inserting metadata into database from {metadata.get('File Name', '')} in folder {metadata.get('Directory', '')}.")
        
        extraction_logger.info(f"Metadata from {metadata.get('File Name', '')} in folder {metadata.get('Directory', '')} extracted and added to the database.")


def start_metadata_extractor():
    start_time = datetime.now()
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

            logger.info(f"Host: {host}, User: {user}, Password: {password}, Database: {database_name}, Table: {table_name}, Image Folder: {image_folder}, Use Yesterday: {use_yesterday}, NSFW: {nsfw}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid configuration: {str(e)}")

        if use_yesterday:
            today = datetime.now()
            yesterday = today - timedelta(days=1)
            formatted_yesterday = yesterday.strftime("%Y-%m-%d")
            image_folder = os.path.join(image_folder, formatted_yesterday)

        # Create a MySQL database and table if it doesn't exist
        conn, existing_columns = connect_database(host, user, password, database_name, table_name, logger)

        # Loop through the images in the folder
        for root, dirs, files in os.walk(image_folder):
            for filename in files:
                if filename.endswith('.png'):
                    image_path = os.path.join(root, filename)
                    metadata = get_image_metadata(image_path, logger)
                    parameters_metadata = metadata.get("parameters", "")
                    extracted_metadata = extract_metadata_from_parameter(parameters_metadata, image_path, nsfw, logger, nsfw_logger)

                    if extracted_metadata is not None:
                        extraction_logger.info(f"Extracted metadata from {image_path} is {extracted_metadata}")
                        insert_metadata_into_database(conn, table_name, existing_columns, extracted_metadata, logger, extraction_logger)

        # Close the database connection
        conn.close()
        end_time = datetime.now()
        time_difference = str(end_time - start_time)
        logger.info(f"Script finished. Duration: {time_difference}")
    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))

if __name__ == "__main__":
    start_metadata_extractor()
