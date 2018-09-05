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

class LinInterpLayer(caffe.Layer):
	def setup(self, bottom, top):
		# check input pair
        	if len(bottom) != 1:
            		raise Exception("Need one input to compute distance.")
		self.batch_size 	= bottom[0].data.shape[0]
		self.num_channel 	= bottom[0].data.shape[1]
		# height and width from stride 2 convolution
		self.height 		= bottom[0].data.shape[2]
		self.width 		= bottom[0].data.shape[3]
		# height and width of the recovered feature map (size is always an even number)
		self.ori_height		= self.height * 2 
		self.ori_width 		= self.width * 2 

	def reshape(self, bottom, top):
		top[0].reshape(self.batch_size, self.num_channel, self.ori_height, self.ori_width)

	def forward(self,bottom,top):
		# exact computation
		top[0].data[:,:,0:self.ori_height:2,0:self.ori_width:2] = bottom[0].data[...]
		# row estimation
		top[0].data[:,:,1:self.ori_height-1:2,0:self.ori_width:2] = (bottom[0].data[:,:,0:self.height-1:1,:] + bottom[0].data[:,:,1:self.height:1,:])/2.0

		# col estimation
		top[0].data[:,:,0:self.ori_height:2,1:self.ori_width-1:2] = (bottom[0].data[:,:,:,0:self.width-1:1] + bottom[0].data[:,:,:,1:self.width:1])/2.0

		# corner estimation
		top[0].data[:,:,1:self.ori_height-1:2,1:self.ori_width-1:2] = (top[0].data[:,:,0:self.ori_height-2:2,1:self.ori_width-1:2] + top[0].data[:,:,2:self.ori_height:2,1:self.ori_width-1:2])/2.0

		# nearest neighbor for border pixels (skipped due to stride 2 conv)
		top[0].data[:,:,:,-1] = top[0].data[:,:,:,-2]
		top[0].data[:,:,-1,:] = top[0].data[:,:,-2,:]


	def backward(self,top,propagate_down,bottom):
		# exact calculation
		bottom[0].diff[...] = top[0].diff[:,:,0:self.ori_height:2, 0:self.ori_width:2]

		# from row estimations
		bottom[0].diff[:,:,:-1,:] +=  0.5*top[0].diff[:,:, 1:self.ori_height-1:2, 0:self.ori_width:2]
		bottom[0].diff[:,:,1:,:]  +=  0.5*top[0].diff[:,:, 1:self.ori_height-1:2, 0:self.ori_width:2]

		# from col estimations
		bottom[0].diff[:,:,:,:-1] +=  0.5*top[0].diff[:,:,0:self.ori_height:2, 1:self.ori_width-1:2]
		bottom[0].diff[:,:,:,1:]  +=  0.5*top[0].diff[:,:,0:self.ori_height:2, 1:self.ori_width-1:2]
		

		# from corner estimations
		corner_interp = top[0].diff[:,:,1:self.ori_height-1:2, 1:self.ori_width-1:2]
		bottom[0].diff[:,:,0:self.height-1:1, 0:self.width-1:1] += 0.25*corner_interp
		bottom[0].diff[:,:,0:self.height-1:1, 1:self.width:1]   += 0.25*corner_interp
		bottom[0].diff[:,:,1:self.height:1,   0:self.width-1:1]   += 0.25*corner_interp
		bottom[0].diff[:,:,1:self.height:1,   1:self.width:1]     += 0.25*corner_interp
		
