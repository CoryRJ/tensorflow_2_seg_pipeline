import numpy as np # linear algebra
import tensorflow as tf

from glob import glob
from Data_handler import Data_handler
from Model_handler import Model_handler
from Experiment_handler import Experiment_handler

print(tf.__version__)
tf.keras.backend.clear_session()

bs = 8
canvas_size = (256, 256)
ps = 3070/canvas_size[0]*.4
np.random.seed(1222)
tf.random.set_seed(1222)

augs = {
    'zoom':(0.8,1.2),
    'rotate':1.0,
    'mirror':'comment out to turn off',
    'chan_shuff':1.0,
    'chan_add':1.0,
    'chan_mult':1.0,
    'coarse_dropout':0.33,
    'blur':.4
    }

run_type = 'train'
if run_type == 'train':
    data_path = './data/dataset001.npy'
    data_h_train = Data_handler(data_path, 'train', equal_sample=False, augs=augs, canvas_size=canvas_size, ps=ps)
    data_h_val = Data_handler(data_path, 'val', augs={}, canvas_size=canvas_size, ps=ps)
    
    weight_path = None
    model = Model_handler(load_path=weight_path)
    
    train = Experiment_handler(model, batch_size=bs, experiment_name='init_setup')
    train.train(1000, data_h_train, data_h_val)
elif run_type == 'test':
    data_path = '../input/hubmap-organ-segmentation/'
    weight_path = None
    model = Model_handler(load_path=weight_path)
    data_h_test = Data_handler(data_path, 'test', augs={}, canvas_size=canvas_size, ps=ps, test_mode=True)

    train = Experiment_handler(model, batch_size=bs, experiment_name='init_setup')
    train.test(data_h_test, 'test')


print('Finished!')