import b2sdk.v2 as b2
import os, sys
from pathlib import Path
import argparse

master_id = ''
master_app_key = ''

# Define the name of the bucket and key that want to create
bucket_name = 'allBatik'
key_name = "allBatik-key"

# Authenticate with Backblaze B2 using master application key
def auth():
    info = b2.InMemoryAccountInfo()
    b2_api = b2.B2Api(info)
    b2_api.authorize_account('production', master_id, master_app_key)
    return b2_api

# Create the specified bucket on B2
def create_bucket(b2_api):
    # Get a name of buckets
    bucket_list_names = [bucket.name for bucket in b2_api.list_buckets()]

    if bucket_name in bucket_list_names:
        ValueError('Your bucket already exists')
    else:
        # Create the bucket
        bucket = b2_api.create_bucket(bucket_name, 'allPublic')
        
        return bucket

# Create new application key with access to spesific bucket
def create_app_key(b2_api, bucket):
    # Get a name of existing keys
    keys = b2_api.list_keys()
    key_names = [key.key_name for key in keys]

    if key_name in key_names:
        ValueError('Your key already exists')
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
        
        content = [
            f'B2_KEY_ID={app_key.id_}',
            f'B2_APPLICATION_KEY={app_key.application_key}',
        ]
        with open('.env', 'a') as f:
            f.write('\n')
            f.write('\n'.join(content))

        local_folder_path = Path('data/batik').resolve()
        remote_folder_path = 'batik'

        for root, _, files in os.walk(local_folder_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                remote_file_path = os.path.join(
                    remote_folder_path,
                    os.path.relpath(local_file_path, local_folder_path)
                ).replace('\\', '/')
        
                print(f'Uploading {local_file_path} to {remote_file_path}')
                uploaded_file = bucket.upload_local_file(local_file_path, remote_file_path)
    
def main():
    parser = argparse.ArgumentParser(description='Upload initial data to B2 bucket')
    parser.add_argument('--id', default=None, help='master key id')
    parser.add_argument('--key', default=None, help='master app key')
    params = parser.parse_args()

    global master_id, master_app_key
    master_id = params.id
    master_app_key = params.key

    b2_api = auth()
    bucket = create_bucket(b2_api)
    create_app_key(b2_api, bucket)

if __name__=='__main__':
    main()