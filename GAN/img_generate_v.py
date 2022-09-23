#generate and balance image from train and test data

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse

import glob
import random
import os

from collections import defaultdict
dic_ls =defaultdict(list)
path_ ="./data/horse2zebra"
for folder in os.listdir(path_):
    for file in os.listdir(os.path.join(path_ +"/" +folder)):
        print(folder, file)
        dic_ls[folder].append(file)
        
test_length = len(dic_ls['testB'])-len(dic_ls['testA'])  
train_length = len(dic_ls['trainB'])-len(dic_ls['trainA'])   

for i in range(test_length):

    mydir ="./data/horse2zebra/testA"
    image_list = random.choice(glob.glob(os.path.join(mydir, '*.jpg')))
    
    # load the input image, convert it to a NumPy array, and then
    # reshape it to have an extra dimension
    print("[INFO] loading example image...")
    image = load_img(image_list)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # construct the image generator for data augmentation then
    # initialize the total number of images generated thus far
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    
    
    total = 0
    
    # construct the actual Python generator
    print("[INFO] generating images...")
    out_ =r"C:\Users\manish.kumar\Desktop\GAN\cycleGAN_horse2zebra\horse2zebra\testA"
    
    imageGen = aug.flow(image, batch_size=1, save_to_dir=out_,
        save_prefix=f"image_{i}", save_format="jpg")
    # loop over examples from our image data augmentation generator
    for image in imageGen:
        # increment our counter
        total += 1
        # if we have reached the specified number of examples, break
        # from the loop
        if total == 1:
            break