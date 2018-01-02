# Spiro Ganas
# 10/2/17
#
# This script uploads local files to an Amazon S3 bucket.
#
# I originally wrote this script to automate the uploading of 140GB of CT Scan images
# that I got from the Kaggle 2017 data science bowl: 
# https://www.kaggle.com/c/data-science-bowl-2017


import boto3
import os


def files_to_upload(source_directory: str) -> list:
    """ Loops over all the files in a local folder.  Returns a link where each 
        element is a tuple where the first element is the full path to a local
        file, and the second element is the s3 key (the folder plus the 
        the filename).
        :param source_directory: The path the the local folder containing the files that will be uploaded.
        :return: A list of tuples.  Each tuple contains a path to the file on the local drive, and
        the AWS S3 key where the file will be stored once it is uploaded.
        """
    upload_file_names = []

    print(source_directory)
    for dirName, subdirList, fileList in os.walk(source_directory):
        for filename in fileList:
            file_path = os.path.join(dirName, filename)
            s3key = os.path.join(os.path.basename(dirName) + '/' + filename)
            upload_file_names.append((file_path, s3key))
    return upload_file_names


def upload_files_to_S3(sourceDir, bucket_name, destDir, aws_access_key_id=None, aws_secret_access_key=None):
    """This function uploads all of the file in sourceDir to the AWS bucket
       bucket_name.  Witin the bucket, the files will be stored in the folder destDir.
       aws_access_key_id and aws_secret_access_key are the codes that you get from the
       AWS IAM website.  You need to grant the user full read/write access to the bucket.
       You can use the AWS CLI and the "aws configure" command to store the password on
       on the computer, eliminating the need to pass it to the python code.
    """

    # set up the connection to the AWS Bucket.
    if aws_access_key_id == None or aws_secret_access_key == None:
        client = boto3.client(service_name='s3', aws_access_key_id=None, aws_secret_access_key=None)
    else:
        client = boto3.client(service_name='s3', aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key)
    transfer = boto3.s3.transfer.S3Transfer(client)

    # Get a list of all the files that have already been uploaded to S3
    MyS3Objects = [s.key for s in boto3.resource('s3').Bucket(bucket_name).objects.filter(Prefix=destDir)]




    uploadFileNames = files_to_upload(sourceDir)

    #print(sourceDir)
    #print(uploadFileNames)


    UploadCounter = 0

    for filename in uploadFileNames:
        sourcepath = filename[0]
        destpath = destDir + '/' + filename[1]

        # If the file is already on S3, don't upload it again
        if destpath in MyS3Objects:
            print(destpath, " is already on S3")
            continue

        UploadCounter += 1
        if UploadCounter % 100 == 0: print("Files Uploaded:", UploadCounter)

        # print ('Uploading %s to Amazon S3 bucket %s' % (sourcepath, bucket_name))

        transfer.upload_file(sourcepath, bucket_name, destpath)

    print("All the files have been uploaded!")


if __name__ == "__main__":
    # Fill in info on data to upload
    # The local folder that will be uploaded
    sourceDir = 'C:\\Users\\spiro\\Desktop\\LungCancer'

    # The name of the AWS S3 bucket that will contain the uploaded files
    bucket_name = 'spiroganas'

    # The folder that will contain the files (note that you leave off the rightmost '/').
    # use the format:  'Folder1/Folder2/Folder3'
    destDir = 'Test_Folder'


    upload_files_to_S3(sourceDir, bucket_name, destDir, aws_access_key_id=None, aws_secret_access_key=None)
