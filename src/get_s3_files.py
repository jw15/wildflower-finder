import boto3
import os
from os import listdir
from os.path import isfile, join

access_key = os.environ['AWS_ACCESS_KEY_ID']
access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

# s3 = boto3.client('s3')
# response = s3.list_buckets()
# Get a list of all bucket names from the response
# buckets = [bucket['Name'] for bucket in response['Buckets']]

def retrieve_from_bucket(file):
    """
    Download matrices from S3 bucket
    """
    conn = boto.connect_s3(access_key, access_secret_key)
    bucket = conn.get_bucket('depression-detect')
    file_key = bucket.get_key(file)
    file_key.get_contents_to_filename(file)
    X = np.load(file)
    return X

def write_to_bucket(bucket_name, filepath, filename):
    s3 = boto3.client('s3')
    s3.upload_file(filepath, bucket_name, filename)
    print('Uploaded - yay!')

def write_folder_to_bucket(root):
    s3 = boto3.client('s3')
    files = [f for f in listdir(root) if isfile(join(root, f))]
    for thing in files:
        write_to_bucket('capstonedatajen', '{}{}'.format(root, thing), '{}'.format(thing))
    print('Uploaded - HUZZAH!')


if __name__ == '__main__':
    # retrieve_from_bucket('imgs_jpgs.zip')
    write_to_bucket('capstonedatajen', '../imgs_for_readme_rsz.zip', 'imgs_for_readme_rsz.zip')
    # write_to_bucket('capstonedatajen', '../imagemagick-identify-parser.zip', 'imagemagick-identify-parser')
