import caffe
import numpy as np

class NegativeLayer(caffe.Layer):
	def setup(self, bottom, top):
		# check input pair
        	if len(bottom) != 1:
            		raise Exception("Need one input only .")
	def reshape(self, bottom, top):
		top[0].reshape(*bottom[0].data.shape)

	def forward(self,bottom,top):
		top[0].data[...] = -bottom[0].data

	def backward(self,top,propagate_down,bottom):
		bottom[0].diff[...] = -top[0].diff

