import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Input

kwargs={
	# 'max_value': 0.5761369157060329, 
	'negative_slope': 0.1,#0.7845179761191806, 
	'threshold': None
	}

layer = tf.keras.layers.ReLU(**kwargs)


input= (np.random.randn(1,32,32,16)).astype(np.float32)
x = Input(batch_shape=input.shape)
y = layer(x)
model = Model(x, y)
pred = model.predict(input)
print(pred)