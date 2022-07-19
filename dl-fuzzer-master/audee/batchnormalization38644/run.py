# import tensorflow as tf
# import numpy as np
# from tensorflow.keras import Model, Input
# import pickle

import os
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Input


# kwargs={
#         'trainable': True, 
#         'dtype': 'float32', 
#         'axis': [2], 
#         'momentum': 0.99, 
#         'epsilon': 0.4022945577533854, 
#         'center': False, 
#         'scale': True, 
#         'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 
#         'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 
#         'moving_mean_initializer': {'class_name': 'Zeros', 
#         'config': {}}, 
#         'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 
#         'beta_regularizer': None, 
#         'gamma_regularizer': None, 
#         'beta_constraint': None, 
#         'gamma_constraint': None}
# layer = tf.keras.layers.BatchNormalization(**kwargs)

input= (np.random.randn(1,32,32,16)).astype(np.float32)
# with open('./input.pkl', 'rb') as f:#input,bug type,params
#         meta = pickle.load(f)

# input1 = meta['input']
# input2 = input1.astype(np.float32)
# input = np.absolute(input2)


# it has to be this model to trigger the bug, not even if with the exact same setting (config)
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(1, 32, 32, 16)]         0
_________________________________________________________________
batch_normalization (BatchNo (1, 32, 32, 16)           96
=================================================================
Total params: 96
Trainable params: 32
Non-trainable params: 64
_________________________________________________________________
'''
model = load_model('./model.h5')
pred = model.predict(input)
print(pred)
