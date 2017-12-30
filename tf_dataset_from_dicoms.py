# Spiro Ganas
# 12/29/17
#
# Download dicom files from AWS S3 and load them into a tf dataset using a generator



import dicom
import numpy as np
from list_images_on_S3 import list_files_on_s3, split_s3_key
from view_ct_scan_on_S3 import download_ct_image_from_s3
import tensorflow as tf
import os


#print('TensorFlow Version: {}'.format(tf.__version__))  # TensorFlow must be version 1.4.0 or greater!



def dicom_generator():
    """Downloads all the dicom files in an AWS S3 bucket and creates a generator
       that yeilds the pixels as an ndarray."""

    # We can't pass arguments to a generator that is feeding a TensorFlow dataset
    bucket = 'lung-cancer-ct-scans'
    folder = 'SampleImages'
    maximum_records = None  #I use this to limit the size of the dataset

    s3_keys = list_files_on_s3(bucket, folder_prefix=folder)

    MyCounter = 1
    for s3_key in s3_keys:
        _, _, filename = split_s3_key(s3_key)
        download_ct_image_from_s3(s3_key, bucket=bucket)
        pixels = dicom.read_file('./temp.dcm').pixel_array
        yield np.reshape(pixels,(512*512))

        if maximum_records is not None and MyCounter > maximum_records: return StopIteration





def dicom_generator_local():
    """Downloads all the dicom files in an AWS S3 bucket and creates a generator
       that yeilds the pixels as an ndarray.
       This generator loops over files stored on the local hard drive"""

    FOLDER = 'C:\\Users\\Administrator\\Desktop\\ct_scans'

    for subdir, dirs, files in os.walk(FOLDER):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = os.path.join(subdir,file)
            #print(filepath)

            pixels = dicom.read_file(filepath).pixel_array
            #print(np.reshape(pixels, (512 * 512)))
            #print(np.shape(pixels))


            #yield np.reshape(pixels, (512 * 512))  #This is a flattened version of the tensor used by autoencoder.py
            yield np.reshape(pixels, [512, 512,1])
















def get_iterator(dicom_generator, batch_size=100, epochs = 100):
    """This takes in an generator and a batch size, and returns a TensorFlow one_shot_iterator object """
    ds = tf.data.Dataset.from_generator(dicom_generator, tf.float32, output_shapes=[512, 512,1])
    #ds = ds.repeat(epochs)  # Create 100 "epochs"
    #ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(batch_size) # Then number of images in a single batch
   # return ds.make_one_shot_iterator()
    return ds.make_initializable_iterator()





if __name__ =='__main__':
    print(dicom_generator_local().__next__())



    #sess = tf.Session()
    #print("Line Break---------------------")
    #x=sess.run(value)
    #print(x ) # (1, array([1]))
    #print(x.shape)
    #print(sess.run(value))  # (2, array([1, 1]))





