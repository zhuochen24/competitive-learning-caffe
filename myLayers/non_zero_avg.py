import caffe
import numpy as np

class NonZeroAvg(caffe.Layer):
	def setup(self, bottom, top):
		# check input pair
		self.epsilon = 0.01
        	if len(bottom) != 1:
            		raise Exception("Need only one input to compute.")
	def reshape(self, bottom, top):
		top[0].reshape(*bottom[0].data.shape)

	def forward(self,bottom,top):
		#shape #batch * 1 * #branch
		top[0].data[...] = np.zeros_like(bottom[0].data, dtype=np.float32)
		#shape: #batch * 1
		self.denominator = np.sum(bottom[0].data, axis=-1) + self.epsilon
		top[0].data[...] = bottom[0].data/self.denominator[:,None]

	def backward(self,top,propagate_down,bottom):
		#shape: #batch * 1 * #branch
		bottom[0].diff[...] = np.zeros_like(top[0].diff, dtype=np.float32)
		self.denominator_sqr = np.square(self.denominator)
		#shape: #batch * 1 * #branch
		self.diff_non_diag = -bottom[0].data/self.denominator_sqr[:,None]
		num_branch = self.diff_non_diag.shape[-1]
		#shape: #batch * #branch * #branch
		self.diff_non_diag_mat = np.repeat(self.diff_non_diag, num_branch, axis=-2) 
		self.diff_diag_mat = np.zeros_like(self.diff_non_diag_mat, dtype=np.float32)
		#shape: #batch * 1
		self.diff_diag = 1.0/self.denominator
		#shape: #batch * #branch
		self.temp_diff_diag_mat = np.repeat(self.diff_diag, num_branch, axis=-1)
		for i in range(self.diff_diag_mat.shape[0]):
			self.diff_diag_mat[i,...] = np.diag(self.temp_diff_diag_mat[i,...])
		gradient = self.diff_diag_mat + self.diff_non_diag_mat
		bottom[0].diff[...] = np.einsum('cnb,cmb->cnm', top[0].diff, gradient)
