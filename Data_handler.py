import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow_addons as tfa
import math
import os
import matplotlib.pyplot as plt
import cv2
import imgaug.augmenters as iaa
import tifffile as tiff
from tqdm import tqdm
from PIL import Image

class Data_handler:
    def __init__(self, data_path, data_type, equal_sample=False, augs={}, canvas_size=(256,256), ps=2.34375, test_mode=False):
        self.data_type = data_type
        self.equal_sample = equal_sample
        self.augs = augs
        self.data_path = data_path
        self.canvas_size = canvas_size
        self.ps = ps
        if not test_mode:
            self.data, self.over_idx = self.load_data(self.data_path)
            self.num_data = len(self.over_idx)

    def process_data(self):
        data = None
        
    def load_data(self, data_path):
        data = None
        return data

    def get_batch(self, batch_size=64):
        batch = []
        if self.equal_sample:
            idx = self.over_idx
        else:
            idx = np.arange(self.num_data,dtype=np.int32)
        end = self.num_data
        np.random.shuffle(idx)
        for val in range(0, end, batch_size):
            #cur_idx = idx[val:val+batch_size]
            batch = [self.data[cur_idx] for cur_idx in idx[val:min(val+batch_size,self.num_data)]]
            yield self.augment_batch(batch)

    def convert_to_canvas(self, img, shape, ps): # what happens if it is zoom OR crop? Might throw an error...
        new_size = math.ceil(shape[0]*ps/self.ps)
        new_size_pad = self.canvas_size[0]*math.ceil(new_size/self.canvas_size[0]) - new_size
        new_size_pad = new_size_pad/2.0
        
        new_size = (new_size, new_size)
        new_size_pad = (math.ceil(new_size_pad), math.floor(new_size_pad))
        #print('new_size:',new_size)
        #print('new_size_pad:',new_size_pad)
        img_n = cv2.resize(img,
                         dsize=new_size,
                         interpolation=cv2.INTER_NEAREST)
        if len(img_n.shape) == 2:
            pad = (new_size_pad,new_size_pad)
        else:
            pad = (new_size_pad,new_size_pad, (0,0))
        img_n_pad = tf.pad(img_n, pad, mode='REFLECT').numpy()
        return img_n_pad

    def convert_from_canvas(self, img, shape, ps): # if channels is 1, this removes it
        img_shape = img.shape[:2]
        new_size = math.ceil(shape[0]*ps/self.ps)
        new_size_pad = self.canvas_size[0]*math.ceil(new_size/self.canvas_size[0]) - new_size
        new_size_pad_low = math.ceil(new_size_pad/2.0)
        new_size_pad_up = img_shape[0] - math.floor(new_size_pad/2.0)
        
        img_n = img[new_size_pad_low:new_size_pad_up,new_size_pad_low:new_size_pad_up,:]
        
        img_n = cv2.resize(img_n,
                         dsize=shape,
                         interpolation=cv2.INTER_NEAREST) # if channels is 1, this removes it
        return img_n
        
        
    def augment_batch(self, batch):
        augs = self.augs
        aug_batch = []
        for instance in batch:
            img = np.copy(instance[0].astype(np.float32))
            seg = np.copy(instance[1].astype(np.float32))
            seg = seg[..., np.newaxis]

            key = instance[2]


            if 'blur' in augs:
                if np.random.random() < augs['blur']:
                    blur = iaa.GaussianBlur(sigma=(0.0, 8.0))
                    img = blur(image=img)
                    #seg = blur(image=seg)
                    
            if 'coarse_dropout' in augs:
                if np.random.random() < augs['coarse_dropout']:
                    drop = iaa.CoarseDropout((0.15, 0.2), size_percent=(0.01, 0.065), per_channel=1.0) #per_channel means it will stay the same or be different for each channel
                    img = drop(image=img)

            img[img > 255] = 255
            img[img < 0] = 0

            if 'rotate' in augs:
                if np.random.random() < augs['rotate']:
                    rot = np.random.uniform(0,2*np.pi)
                    img = tf.keras.layers.RandomRotation((rot, rot))(img)
                    seg = tf.keras.layers.RandomRotation((rot, rot))(seg)

            if 'zoom' in augs:
                zoom = np.random.uniform(augs['zoom'][0],augs['zoom'][1])
                transform = np.array([1.0*zoom, .0, (1.-zoom)*(self.canvas_size[0]/2),
                                      .0, 1.0*zoom, (1.-zoom)*(self.canvas_size[0]/2), 
                                      .0, .0], dtype=np.float32)
                                      
                img = tfa.image.transform(img, transform, fill_mode='reflect')
                seg = tfa.image.transform(seg, transform, fill_mode='reflect')

            if 'mirror' in augs:
                if np.random.random() < 0.5:
                    img = tf.image.flip_left_right(img)
                    seg = tf.image.flip_left_right(seg)




            img = img/255.0
            seg = seg[:,:,0]
            
            aug_batch += [(img,seg,key)]
            #aug_batch += [(img,seg)]
        return aug_batch