import caffe
import numpy as np

class NegThreshold(caffe.Layer):
	def setup(self, bottom, top):
		# check input pair
        	if len(bottom) != 1:
            		raise Exception("Need only one input to compute.")
	def reshape(self, bottom, top):
		top[0].reshape(*bottom[0].data.shape)

	def forward(self,bottom,top):
		#top[0].data[...] = np.tensordot(bottom[0].data, bottom[1].data,axes=([],[]))
		top[0].data[...] = np.zeros_like(bottom[0].data, dtype=np.float32) - 10.0
		top[0].data[bottom[0].data > 0.7] = 1.0

	def backward(self,top,propagate_down,bottom):
		bottom[0].diff[...] = top[0].diff
		bottom[0].diff[bottom[0].data <= 0.7] = 0.0
