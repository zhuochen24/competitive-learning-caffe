import caffe
import numpy as np

class StochOutLayer(caffe.Layer):
	def setup(self, bottom, top):
		# check input pair
        	if len(bottom) != 2:
            		raise Exception("Need two inputs to compute distance.")
	def reshape(self, bottom, top):
		# bottom[0] is branch output
		# bottom[1] is selecting prob
# for mnist,cifar10
		#top[0].reshape(bottom[0].data.shape[0],10)
# for IMAGENET
		top[0].reshape(bottom[0].data.shape[0],1000)

	def forward(self,bottom,top):
		best_choice = np.argmax(bottom[1].data,axis=1)
		for ind, best in enumerate(best_choice):
			top[0].data[ind,:] = bottom[0].data[ind,best,:]
		#print 'selecting prob:',bottom[1].data
		#print 'stochOut_top0:',top[0].data[...]

	def backward(self,top,propagate_down,bottom):
		pass

