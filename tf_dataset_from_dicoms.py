# Spiro Ganas
# 12/29/17
#
# Download dicom files from AWS S3 and load them into a tf dataset using a generator



import dicom
import numpy as np
from list_images_on_S3 import list_files_on_s3, split_s3_key
from view_ct_scan_on_S3 import download_ct_image_from_s3
from tensorflow import Dataset


def dicom_generator(bucket = 'lung-cancer-ct-scans',folder = 'SampleImages'):
    """Downloads all the dicom files in an AWS S3 bucket and creates a generator
       that yeilds the pixels as an ndarray."""

    s3_keys = list_files_on_s3(bucket, folder_prefix=folder)

    MyCounter = 1
    for s3_key in s3_keys:
        _, _, filename = split_s3_key(s3_key)
        download_ct_image_from_s3(s3_key, bucket=bucket)
        yield dicom.read_file('./temp.dcm').pixel_array



ds = Dataset.from_generator(
    dicom_generator, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))

value = ds.make_one_shot_iterator().get_next()