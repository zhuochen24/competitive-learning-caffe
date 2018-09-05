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
		self.height 		= bottom[0].data.shape[2]
		self.width 		= bottom[0].data.shape[3]

	def reshape(self, bottom, top):
		top[0].reshape(*bottom[0].data.shape)

	def forward(self,bottom,top):
		top[0].data[...] = bottom[0].data[...]
		rows_selected = np.arange(0,self.height,2)
		computed = top[0].data[:,:,rows_selected,:]
		#print computed.shape
		#print np.arange(0,rows_selected.shape[0]-1,1).shape
		#print np.arange(1,rows_selected.shape[0],1).shape
		#print computed[:,:,np.arange(0,rows_selected.shape[0]-1,1),:].shape
		#print computed[:,:,np.arange(1,rows_selected.shape[0],1),:].shape
		#print top[0].data[:,:,np.arange(1,self.height,2),:].shape
		top[0].data[:,:,np.arange(1,self.height-1,2),:] = (computed[:,:,np.arange(0,rows_selected.shape[0]-1,1),:] + computed[:,:,np.arange(1,rows_selected.shape[0],1),:])/2.0

		cols_selected = np.arange(0,self.width,2)
		computed = top[0].data[:,:,:,cols_selected]
		top[0].data[:,:,:,np.arange(1,self.width-1,2)] = (computed[:,:,:,np.arange(0,cols_selected.shape[0]-1,1)] + computed[:,:,:,np.arange(1,cols_selected.shape[0],1)])/2.0


	def backward(self,top,propagate_down,bottom):
		bottom[0].diff[...] = top[0].diff
		rows_selected = np.arange(0,self.height,2)
		rows_interp   = np.arange(1,self.height-1,2)
		bottom[0].diff[:,:,rows_selected[:-1],:] +=  0.5*top[0].diff[:,:,rows_interp,:]
		bottom[0].diff[:,:,rows_selected[1:],:]  +=  0.5*top[0].diff[:,:,rows_interp,:]

		cols_selected = np.arange(0,self.width,2)
		cols_interp = np.arange(1,self.width-1,2)
		bottom[0].diff[:,:,:,cols_selected[:-1]] +=  0.5*top[0].diff[:,:,:,cols_interp]
		bottom[0].diff[:,:,:,cols_selected[1:]]  +=  0.5*top[0].diff[:,:,:,cols_interp]
		

		corner_interp = top[0].diff[:,:,1:self.height-1:2,1:self.width-1:2]
		bottom[0].diff[:,:,0:self.height-2:2,0:self.width-2:2] += 0.25*corner_interp
		bottom[0].diff[:,:,0:self.height-2:2,2:self.width:2]   += 0.25*corner_interp
		bottom[0].diff[:,:,2:self.height:2,0:self.width-2:2]   += 0.25*corner_interp
		bottom[0].diff[:,:,2:self.height:2,2:self.width:2]     += 0.25*corner_interp
		
		bottom[0].diff[:,:,1:self.height-1:2,0:self.width:2]   = 0
		bottom[0].diff[:,:,0:self.height:2,1:self.width-1:2]   = 0
		bottom[0].diff[:,:,1:self.height-1:2,1:self.width-1:2] = 0
