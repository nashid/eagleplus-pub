import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Input


kwargs={
	'filters': 19, 
	'kernel_size': (3, 3), 
	'padding': 'same', 
	'output_padding': (0, 0), 
	'strides': 2, 
	'dilation_rate': (1, 2), 
	'data_format': 'channels_last'
	}
	
layer = tf.keras.layers.Conv2DTranspose(**kwargs)

input= (np.random.randn(1,32,32,16)).astype(np.float32)
x = Input(batch_shape=input.shape)
y = layer(x)
model = Model(x, y)
pred = model.predict(input)
print(pred)