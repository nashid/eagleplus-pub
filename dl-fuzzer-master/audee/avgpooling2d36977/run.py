import numpy as np
import torch
   
kwargs = {
	'ceil_mode': True, 
	# 'count_include_pad': False, 
	'kernel_size': (1, 2), 
	# 'padding': (0, 1), 
	'stride': (2, 2)
	}  
	
layer = torch.nn.AvgPool2d(**kwargs)

input = (10 * np.random.randn(1,16,4,4)).astype(np.float32)   
out = layer(torch.tensor(input))
# modelinput=torch.from_numpy(input)
# output = layer(modelinput)
# ytrue=output.detach().numpy()
nanresult=np.isnan(out).any()

print(nanresult)