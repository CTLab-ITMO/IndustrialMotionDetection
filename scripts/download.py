import os
import sys
import argparse
import requests
import dotenv
import subprocess

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.join(project_dir, 'src'))

from logger import Logger

def download_kaggle_dataset(dataset_id, output_file_path):
    SHOW_LOG = True
    logger = Logger(SHOW_LOG).get_logger(__name__)
    
    env_filename = '.env'
    
    dotenv_path = os.path.join(project_dir, env_filename)

    if not os.path.exists(dotenv_path):
        logger.error(f"Error: {dotenv_path} file not found")
        sys.exit()
    
    dotenv.load_dotenv(dotenv_path)

    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    logger.info(f"Downloading: {args.dataset} → {args.output}")
    
    try:
        upload_command = f"curl -L -u {kaggle_username}:{kaggle_key} -o {output_file_path} https://www.kaggle.com/api/v1/datasets/download/{dataset_id}"
        subprocess.run(upload_command, shell=True, check=True)
        logger.info("Dataset download initiated successfully!")
    except Exception as e:
        logger.error(f'Dataset download failed: {e}')
        sys.exit()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Download dataset from Kaggle')
    parser.add_argument('-d', '--dataset', 
                        required=True, 
                        help='Kaggle dataset identifier')
    parser.add_argument('-o', '--output', 
                        required=True, 
                        help='Output file path')
    args = parser.parse_args()
    download_kaggle_dataset(
        dataset_id=args.dataset,
        output_file_path=args.output)
