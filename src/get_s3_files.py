import boto

access_key = os.environ['AWS_ACCESS_KEY_ID']
access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

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

def write_to_bucket(bucket_name, file):
    conn = boto.connect_s3(access_key, access_secret_key)
    b = conn.get_bucket(bucket_name)
    file_object = b.new_key('sample.txt')

if __name__ == '__main__':
    retrieve_from_bucket('imgs_jpgs.zip')
    write_to_bucket('lifeclef2015jen', )
