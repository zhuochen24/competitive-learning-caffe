import caffe
import numpy as np

class MatvecLayer(caffe.Layer):
	def setup(self, bottom, top):
		# check input pair
        	if len(bottom) != 2:
            		raise Exception("Need two inputs to compute distance.")
	def reshape(self, bottom, top):
		top[0].reshape(bottom[0].data.shape[0],bottom[0].data.shape[1],bottom[1].data.shape[2])

	def forward(self,bottom,top):
		#top[0].data[...] = np.tensordot(bottom[0].data, bottom[1].data,axes=([],[]))
		top[0].data[...] = np.einsum('cnb,cbm->cnm', bottom[0].data, bottom[1].data)

	def backward(self,top,propagate_down,bottom):
		bottom[0].diff[...] = np.einsum('cnb,cmb->cnm',top[0].diff,bottom[1].data)
		#bottom[0].diff.reshape(*bottom[0].data.shape)
		bottom[1].diff[...] = np.einsum('cbn,cbm->cnm',bottom[0].data,top[0].diff)
