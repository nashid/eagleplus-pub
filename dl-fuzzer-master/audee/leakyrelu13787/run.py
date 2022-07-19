import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Input


kwargs = {'alpha': None}
layer = tf.keras.layers.LeakyReLU(**kwargs)

input= (np.random.randn(1,32,32,16)).astype(np.float32)
# x = Input(batch_shape=input.shape)
# y = layer(x)
# model = Model(x, y)
# pred = model.predict(input)
# print(pred)
out = layer(input)
print(np.isnan(ytrue).any())
