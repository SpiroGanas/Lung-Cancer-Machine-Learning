# Spiro Ganas
# 9/27/17
#
# Python 3 script to


######### NOTES ########################################
# 1.  Each "slice" is a 512x512 image, stored in a single .dcm file.
# 2.  A 3-dimensional CT Scan consists of between 94 and 541 slices (according to the Stage 1 data).
# 3.  We need to rescale all the CT scans so they have the same number of slices.
#########################################################




####### CONSTANTS ##########################

# This folder contains one subfolder per patient
data_dir = 'D:\\Lung Cancer Dataset\\stage1'

# This CSV lists all patients and shows which ones have canver
truth_file = 'D:\\Lung Cancer Dataset\\stage1_labels.csv'

# These values are used to downscale the images to a uniform size
IMAGE_DIMESNION = 512  # Downscale the images so the size of a slice is IMAGE_DIMENSIONxIMAGE_DIMENSION
NUMBER_OF_SLICES =20   # A patient can have between X and Y slices.  This downscales all patients to the same number of slices.


#############################################




import dicom #http://pydicom.readthedocs.io/en/stable/getting_started.html
#print("pydicom version: ", dicom.__version__)



import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf


# Change this to wherever you are storing your data:
# IF YOU ARE FOLLOWING ON KAGGLE, YOU CAN ONLY PLAY WITH THE SAMPLE DATA, WHICH IS MUCH SMALLER


labels_df = pd.read_csv(truth_file, index_col=0)


#patients = list(labels_df.index.values)
patients = os.listdir(data_dir)


print(labels_df.head())
print(patients)
print(len(patients))


temp_min = 9999
temp_max = 0

for patient in patients:
    try:
        label = labels_df.get_value(patient, 'cancer')
        path = data_dir +'/'+ patient

        # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
        print(slices[0].pixel_array.shape, len(slices))
        if len(slices)>temp_max: temp_max=len(slices)
        if len(slices) < temp_min: temp_min = len(slices)
    except:
        pass


print("Minimum number of slices:", temp_min)
print("Maximum number of slices:", temp_max)

