# This plots a CT scan image
#
# I used this code to learn how to import images from Amazon S3

import boto3
import dicom
import matplotlib.pyplot as plt


# This loads the image from the local hard drive
#image = 'D:\\Lung Cancer Dataset\\stage1\\00cba091fa4ad62cc3200a657aeb957e\\0a291d1b12b86213d813e3796f14b329.dcm'



######## Start of boto3 code ############################
import boto3
import botocore


# I had to follow these steps:
# 1.  Set up an IAM user that has "programatic access".
# 2.  from cmd, run:  aws configure
# 3.  Enter the access key ID and Secret Access Key, which are availible from the IAM user page on the AWS website.


def view_local_ct_image(filename: str ='./temp.dcm'):
    """Opens a dicom file and displays it using matplotlib."""
    ct_slice = dicom.read_file(filename)
    plt.imshow(ct_slice.pixel_array)
    plt.show()


def download_ct_image_from_s3(s3_key: str, bucket: str = 'lung-cancer-ct-scans'):
    """Download a file and temporarily save it to disk."""
    s3 = boto3.resource('s3')
    # I'm temporarily storing the dcm file to disk.
    # This is because pydicom expects a file, not a python object
    s3.meta.client.download_file(bucket, s3_key, './temp.dcm')


def view_s3_ct_scan(s3_key: str, bucket: str = 'lung-cancer-ct-scans'):
    """Downloads and then views a CT Scan that is stored on AWS S3."""
    download_ct_image_from_s3(s3_key, bucket)
    view_local_ct_image(filename='./temp.dcm')






if __name__ =="__main__":
    BUCKET = 'lung-cancer-ct-scans' # replace with your bucket name
    KEY = 'SampleImages/0de72529c30fe642bc60dcb75c87f6bd/fbbb43b670ef83964e57416bbdfeafb0.dcm' # replace with your object key
    view_s3_ct_scan(KEY, BUCKET)


