import caffe
import numpy as np

#class StochLossLayer(caffe.Layer):
#	def setup(self, bottom, top):
#		# check input pair
#        	if len(bottom) != 2:
#            		raise Exception("Need two inputs to compute distance.")
#	def reshape(self, bottom, top):
#		# bottom[0] is euclinean distance
#		# bottom[1] is selecting prob
#		top[0].reshape(bottom[0].data.shape[0],1)
#
#	def forward(self,bottom,top):
#		#top[0].data[...] = np.einsum('cnb,cbm->cnm', bottom[0].data, bottom[1].data)
#		top[0].data[...] =-np.log(np.sum(np.exp(-0.5*bottom[0].data)*bottom[1].data, axis=1))[:,None]
#
#	def backward(self,top,propagate_down,bottom):
#		bottom[0].diff[...] = bottom[1].data*np.exp(-0.5*bottom[0].data)*np.sqrt(bottom[0].data)/np.sum(bottom[1].data*np.exp(-0.5*bottom[0].data),axis=1)[:,None]
#		bottom[1].diff[...] = -np.exp(-0.5*bottom[0].data)/np.sum(bottom[1].data*np.exp(-0.5*bottom[0].data), axis=1)[:,None]

class StochLossLayer(caffe.Layer):
	def setup(self, bottom, top):
		# check input pair
        	if len(bottom) != 3:
            		raise Exception("Need three inputs to compute distance.")
		self.batch_size = bottom[0].data.shape[0]
		self.num_branch = bottom[0].data.shape[1]
		self.num_class = bottom[0].data.shape[2]
		self.gamma = 1
		#self.label = np.zeros((self.batch_size,self.num_class))
		#self.label1d = bottom[1].data.astype(int)
		#self.label[np.arange(self.batch_size),self.label1d] = 1 
		#self.label_3d = np.tile(self.label[:,None,:],(1,self.num_branch,1))
		#self.eucDist = np.sum(np.square(self.label_3d-bottom[0].data), axis=2)


	def reshape(self, bottom, top):
		# bottom[0] is prediction: 3D batch,branch,element
		# bottom[1] is label: 1D batch
		# bottom[2] is selecting prob: 2D batch,branch
		top[0].reshape(bottom[0].data.shape[0],1)

	def forward(self,bottom,top):
		label = np.zeros((self.batch_size,self.num_class))
		label1d = bottom[1].data.astype(int)
		label[np.arange(self.batch_size),label1d] = 1 
		label_3d = np.tile(label[:,None,:],(1,self.num_branch,1))
		eucDist = np.sum(np.square(label_3d-bottom[0].data), axis=2)
		top[0].data[...] =-np.log(np.sum(np.exp(-0.5*self.gamma*eucDist)*bottom[2].data, axis=1))[:,None]

		#print 'label 1d:', bottom[1].data
		#print 'label 1d type:', type(bottom[1].data)
		#print 'label 1d mylabel:', label1d
		#print 'label 2d:', label[...]
		#print 'label 3d:', label_3d[...]
		#print 'EucDist:',eucDist[...]
		#print 'stochLoss_top:',top[0].data[...]
		#print 'pred:', bottom[0].data[...]

	def backward(self,top,propagate_down,bottom):
		label = np.zeros((self.batch_size,self.num_class))
		label1d = bottom[1].data.astype(int)
		label[np.arange(self.batch_size),label1d] = 1 
		label_3d = np.tile(label[:,None,:],(1,self.num_branch,1))
		eucDist = np.sum(np.square(label_3d-bottom[0].data), axis=2)

		softmax_partOutput = (bottom[2].data*np.exp(-0.5*self.gamma*eucDist)/np.tile(np.sum(bottom[2].data*np.exp(-0.5*self.gamma*eucDist),axis=1)[:,None],(1,self.num_branch)))
		softmax_partOutput_aug = np.tile(softmax_partOutput[...,None],(1,1,self.num_class))
		top_diff_aug = np.tile(top[0].diff,(1,bottom[0].data.shape[1]))
		top_diff_aug = np.tile(top_diff_aug[...,None],(1,1,bottom[0].data.shape[2]))
		bottom[0].diff[...] = top_diff_aug * softmax_partOutput_aug *self.gamma* (bottom[0].data-label_3d)

		bottom[2].diff[...] = np.tile(top[0].diff,(1,self.num_branch))*(-(np.exp(-0.5*self.gamma*eucDist)/np.tile(np.sum(bottom[2].data*np.exp(-0.5*self.gamma*eucDist),axis=1)[:,None],(1,self.num_branch))))

		#print 'stochLoss_bottom0:',bottom[0].diff[...]
		#print 'stochLoss_bottom2:',bottom[2].diff[...]
