# Spiro Ganas
# 12/29/17
#
# Download dicom files from AWS S3 and save them as jpgs


import dicom
import numpy as np
from AWS_S3_Programs.list_images_on_S3 import list_files_on_s3, split_s3_key
from AWS_S3_Programs.view_ct_scan_on_S3 import download_ct_image_from_s3


def download_and_convert_dicoms(bucket = 'lung-cancer-ct-scans',folder = 'SampleImages', local_folder = './data' ):
    """Downloads all the dicom files in an AWS S3 bucket and saves them locally as a csv file"""
    s3_keys = list_files_on_s3(bucket, folder_prefix=folder)
    #print(s3_keys)
    MyCounter = 1
    for s3_key in s3_keys:
        _,_,filename = split_s3_key(s3_key)
        download_ct_image_from_s3(s3_key, bucket=bucket)
        ct_slice = dicom.read_file('./temp.dcm').pixel_array

        save_location = './data/' + filename + '.txt'
        np.savetxt(save_location, ct_slice)

        print("Current Image:", MyCounter)
        MyCounter+=1
        if MyCounter>1000: exit()





if __name__ =="__main__":
    download_and_convert_dicoms(bucket = 'lung-cancer-ct-scans',folder = 'SampleImages', local_folder = './data')