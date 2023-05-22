#!/usr/bin/env python3

""" A python script for working with Backblaze B2 """

import b2sdk.v2 as b2
import os, sys, json
import argparse

master_id = ''
master_app_key = ''
mlflow_pass = ''

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
