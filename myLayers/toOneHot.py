import caffe
import numpy as np

class ToOneHotLayer(caffe.Layer):
        def setup(self, bottom, top):
                # check input pair
                if len(bottom) != 1:
                        raise Exception("Need one inputs to compute distance.")
                self.batch_size = bottom[0].data.shape[0]
		self.num_class = 1000
                #self.label = np.zeros((self.batch_size,self.num_class))
                #self.label1d = bottom[1].data.astype(int)
                #self.label[np.arange(self.batch_size),self.label1d] = 1
                #self.label_3d = np.tile(self.label[:,None,:],(1,self.num_branch,1))
                #self.eucDist = np.sum(np.square(self.label_3d-bottom[0].data), axis=2)


        def reshape(self, bottom, top):
                top[0].reshape(self.batch_size,self.num_class)

        def forward(self,bottom,top):
                label = np.zeros((self.batch_size,self.num_class))
                label1d = bottom[0].data.astype(int)
                label[np.arange(self.batch_size),label1d] = 1
                #label_3d = np.tile(label[:,None,:],(1,self.num_branch,1))
                #eucDist = np.sum(np.square(label_3d-bottom[0].data), axis=2)
                #top[0].data[...] =-np.log(np.sum(np.exp(-0.5*self.gamma*eucDist)*bottom[2].data, axis=1))[:,None]
		top[0].data[...] = label

                #print 'label 1d:', bottom[1].data
                #print 'label 1d type:', type(bottom[1].data)
                #print 'label 1d mylabel:', label1d
                #print 'label 2d:', label[...]
                #print 'label 3d:', label_3d[...]
                #print 'EucDist:',eucDist[...]
                #print 'stochLoss_top:',top[0].data[...]
                #print 'pred:', bottom[0].data[...]

        def backward(self,top,propagate_down,bottom):
		pass
