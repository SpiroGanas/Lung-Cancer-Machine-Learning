# Spiro Ganas
# 12/27/17
#
# Lists all the files in the given S3 bucket/folder
# I use this to get a list of all the CT scan images


# I had to follow these steps to set up authentication using the AWS CLI program:
# 1.  Set up an IAM user that has "programatic access".
# 2.  from cmd, run:  aws configure
# 3.  Enter the access key ID and Secret Access Key, which are availible from the IAM user page on the AWS website.


import boto3


def list_files_on_s3(bucket_name, folder_prefix = 'SampleImages'):
    """Returns a list of all the files in this bucket or folder"""
    list_of_files = []
    s3 = boto3.resource('s3')

    for obj in s3.Bucket(bucket_name).objects.filter(Prefix=folder_prefix):
        list_of_files.append(obj.key)
    return list_of_files


def split_s3_key(s3_key: str) -> "tuple containing folder, patient ID and image filename":
    """Splits the S3 object key into a tuple containing the folder, the patient ID and the image filename"""
    folder = s3_key[:s3_key.find("/")]
    patient = s3_key[s3_key.find("/")+1:s3_key.find("/", s3_key.find("/")+1)]
    image = s3_key[s3_key.find("/", s3_key.find("/")+1)+1:]
    return folder, patient, image







if __name__ == '__main__':
    BUCKET = 'lung-cancer-ct-scans'
    FOLDER = 'SampleImages'
    s3_keys = list_files_on_s3(BUCKET, folder_prefix=FOLDER)
    print("Number of files: {}".format(len(s3_keys)))
    print(s3_keys)
    print(split_s3_key(s3_keys[0]))
