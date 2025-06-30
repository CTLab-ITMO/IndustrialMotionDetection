import os
import json
import subprocess
import argparse
import dotenv
import sys

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.join(project_dir, 'src'))

from config import YamlConfigReader
from logger import Logger


def create_kaggle_dataset(dataset_name):
    SHOW_LOG = True
    logger = Logger(SHOW_LOG).get_logger(__name__)
    
    env_filename = '.env'
    preproc_conf_filename = 'conf/meva_preproc.yaml'
    
    dotenv_path = os.path.join(project_dir, env_filename)
    
    if not os.path.exists(dotenv_path):
        logger.error(f"Error: {dotenv_path} file not found")
        sys.exit()
    
    dotenv.load_dotenv(dotenv_path)
    
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    
    preproc_conf_path = os.path.join(project_dir, preproc_conf_filename)
    
    if not os.path.exists(preproc_conf_path):
        logger.error(f"Error: {preproc_conf_path} file not found")
        sys.exit()
    
    config_source = YamlConfigReader(preproc_conf_path)
    params = config_source.get_all()
    logger.info(f"Preproc conf file params {params}")
    
    result_folder = os.path.join(project_dir, params['result_folder'])
    
    if not os.path.isdir(result_folder):
        logger.error(f"Error: {result_folder} dir not found")
        sys.exit()

    try:
        send_folder_path = os.path.join(result_folder, os.pardir, 'meva_send')
        os.makedirs(send_folder_path, exist_ok=True)
        logger.info(f"Created folder for kaggle: {send_folder_path}")

        # Zip send folder path
        processed_zip_path = f"{send_folder_path}/meva-processed.zip"
        if not os.path.exists(processed_zip_path):            
            zip_command = f"cd {result_folder} && zip -r {processed_zip_path} ."
            subprocess.run(zip_command, shell=True, check=True)
            logger.info(f"Zipped processed folder {processed_zip_path}")
        
        # Create dataset-metadata.json
        metadata = {
            "title": dataset_name,
            "id": f"{kaggle_username}/{dataset_name}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        metadata_filename = "dataset-metadata.json"
        
        metadata_path = os.path.join(send_folder_path, metadata_filename)
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Created metadata file: {metadata_path}")
        
        # Upload to Kaggle using kaggle api
        upload_command = f"kaggle datasets create -u -p {send_folder_path} --dir-mode zip"
        subprocess.run(upload_command, shell=True, check=True)
        logger.info("Dataset upload initiated successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessed MEVA uploading script")
    
    parser.add_argument("-dn", "--dataset-name",
                        type=str,
                        help="Sepcify dataset name",
                        required=True,
                        nargs="?")

    args = parser.parse_args()
    
    create_kaggle_dataset(dataset_name=args.dataset_name)
