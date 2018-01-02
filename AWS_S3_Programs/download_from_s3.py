# Spiro Ganas
# 1/2/18
#
# Downloads a folder or bucket from AWS S3


import boto3
import os.path
from AWS_S3_Programs.list_images_on_S3 import list_files_on_s3, split_s3_key




def download_ct_image_from_s3(s3_key: str, bucket: str = 'lung-cancer-ct-scans', local_folder: str = './temp.dcm'):
    """Download a file and temporarily save it to disk."""
    s3 = boto3.resource('s3')
    # I'm temporarily storing the dcm file to disk.
    # This is because pydicom expects a file, not a python object
    s3.meta.client.download_file(bucket, s3_key, local_folder)




def download_folder_from_s3(bucket: str ='', folder: str ='', local_folder: str = ''):
    '''Downloads a folder or an entire bucket from AWS S2 and saves the files in local_folder'''
    list_of_s3_keys = list_files_on_s3(bucket, folder)
    for ct_image in list_of_s3_keys:
        folder, patient, image = split_s3_key(ct_image)

        # Create the directories if they don't already exist
        if not os.path.exists(os.path.join(local_folder)):os.makedirs(os.path.join(local_folder))
        if not os.path.exists(os.path.join(local_folder,folder)): os.makedirs(os.path.join(local_folder,folder))
        if not os.path.exists(os.path.join(local_folder,folder,patient)): os.makedirs(os.path.join(local_folder,folder,patient))

        save_path = os.path.join(local_folder, folder, patient, image)
        download_ct_image_from_s3(s3_key=ct_image, bucket=bucket, local_folder=save_path)

    print("Files have been downloaded and are located here:" + local_folder)





if __name__ == '__main__':
    BUCKET = 'lung-cancer-ct-scans'
    FOLDER = 'SampleImages'
    s3_keys = download_folder_from_s3(bucket=BUCKET, folder=FOLDER, local_folder='..\\data\\S3_Downloads\\')
