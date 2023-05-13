#!/usr/bin/env python3

""" A python script for working with Backblaze B2 """

import b2sdk.v2 as b2
import os, sys, json
import argparse

master_id = input('Enter your B2 account id: ')
master_app_key = input('Enter your B2 application key: ')
mlflow_pass = input('Enter your MLFlow pass/token: ')

# Define the name of the bucket and key that want to create
bucket_name = 'allBatik'
key_name = "allBatik-key"

# Authenticate with Backblaze B2 using master application key
def auth():
    info = b2.InMemoryAccountInfo()
    b2_api = b2.B2Api(info)
    b2_api.authorize_account('production', master_id, master_app_key)
    return info, b2_api

# Create the specified bucket on B2
def create_load_bucket(b2_api):
    # Get a name of buckets
    bucket_list_names = [bucket.name for bucket in b2_api.list_buckets()]

    if bucket_name in bucket_list_names:
        print('Your bucket already exists')
        bucket = b2_api.get_bucket_by_name(bucket_name)
    else:
        # Create the bucket
        bucket = b2_api.create_bucket(bucket_name, 'allPublic')

    return bucket

# Create new application key with access to spesific bucket
def create_load_app_key(b2_api, bucket):
    # Get a name of existing keys
    keys = b2_api.list_keys()
    key_names = [key.key_name for key in keys]

    file_name='id-key.json'

    if key_name in key_names:
        print('Your key already exists')
        downloaded_file = bucket.download_file_by_name(file_name)
        downloaded_file.save_to(file_name, 'wb+')
        with open(file_name, 'rb') as file:
            content = json.load(file)
        os.remove(file_name)
        
    else:
        # Define the capabilities for the new application key
        capabilities = ['deleteFiles', 'listBuckets', 'listFiles', 'readBucketEncryption', 
                        'readBucketReplications', 'readBuckets', 'readFiles', 'shareFiles', 
                        'writeBucketEncryption', 'writeBucketReplications', 'writeFiles']
        # Create the new application key
        try:
            app_key = b2_api.create_key(key_name=key_name, capabilities=capabilities, bucket_id=bucket.get_id())
        except b2.exception.B2Error as e:
            sys.exit(f"Error creating application key: {e}")
        
        # Upload id and keys in bucket
        content = {
            'id':app_key.id_,
            'key':app_key.application_key,
        }
        file_data = json.dumps(content).encode()
        bucket.upload_bytes(file_data, file_name, content_type='application/json')
    
    return content.values()

# Write .env file
def write_dotenv():
    info, b2_api = auth()
    bucket = create_load_bucket(b2_api)
    key_id, application_key = create_load_app_key(b2_api, bucket)

    with open(".env", "w") as f:
        # Write the key ID, application key and ect
        f.write(f"B2_KEY_ID={key_id}\n")
        f.write(f"B2_APPLICATION_KEY={application_key}\n")

        # the endpoint URL for the bucket
        endpoint_url = info.get_s3_api_url()
        f.write(f"B2_URL_ENDPOINT={endpoint_url}\n")
        # S3 URL format
        remote_url = f's3://{bucket_name}/initial_data'
        f.write(f"B2_URL_REMOTE={remote_url}\n")

        # MLFlow tracking uri, username, and password/token
        mlflow_username = 'afhabibieee'
        tracking_uri = f'https://dagshub.com/{mlflow_username}/fsl-batik.mlflow'
        f.write(f"MLFLOW_TRACKING_URI={tracking_uri}\n")
        f.write(f"MLFLOW_TRACKING_USERNAME={mlflow_username}\n")
        f.write(f"MLFLOW_TRACKING_PASSWORD={mlflow_pass}\n")
    
    os.system("echo '.env' >> .gitignore")
    return key_id, application_key, endpoint_url, remote_url

# Init DVC
def init_dvc():
    key_id, application_key, endpoint_url, remote_url = write_dotenv()
    
    if os.path.exists('.dvc'):
        os.system('rm -f .dvc/config')
        os.system('dvc init -f')
    else:
        os.system('dvc init')

    # Configure the remote with DVC
    os.system(f'dvc remote add -f -d myremote {remote_url}')
    os.system(f'dvc remote modify myremote endpointurl {endpoint_url}')
    os.system(f'dvc remote modify myremote --local access_key_id {key_id}')
    os.system(f'dvc remote modify myremote --local secret_access_key {application_key}')

def pull_dvc():
    key_id, application_key, _, _ = write_dotenv()

    # DVC user configuration
    os.system(f'dvc remote modify myremote --local access_key_id {key_id}')
    os.system(f'dvc remote modify myremote --local secret_access_key {application_key}')
    
    os.system('dvc pull -r myremote')
    # Make sure that all files were pulled
    os.system('dvc pull -r myremote')

def main():
    parser = argparse.ArgumentParser(description='Init or Pull DVC data and Configure MLFlow')
    parser.add_argument('--mode', default=None, help='init/pull')
    params = parser.parse_args()
    if params.mode == 'init':
        init_dvc()
    elif params.mode == 'pull':
        pull_dvc()
    else:
        ValueError('The arg entered is not available')

if __name__=='__main__':
    main()