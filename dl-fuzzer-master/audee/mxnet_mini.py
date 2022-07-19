from mxnet.gluon import nn
import mxnet as mx
import numpy as np

#layer = nn.MaxPool3D()
layer = nn.Dense(32)
#init = mx.initializer.Uniform()
layer.initialize()
input = mx.nd.array((np.random.randn(1,1,8,8,16)).astype(np.float32))
output = layer(input)
print(output)

# some layer requires param initialization, e.g. dense



# produce inf:
# a = mxnet.nd.array([4022, 10432, 13904, 6360, 10128, 12752, 8616, 10536, 12200, -5452, 1838, 5432, -2684, 1774, 4476, 198.625, 2700, 4052, -8264, -1236, 2524, -5188, -880.5, 1629, -2280, 151.875, 1655], dtype=float16)
# a.sum()