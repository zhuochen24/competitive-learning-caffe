import numpy as np
#import sklearn.cluster as cls

def save_mat(filename,mat):
	with open(filename,'w') as f:
		np.savetxt(f,mat,delimiter=',')


clusters = np.genfromtxt('./clustering/9cluster_oneLevel.csv', delimiter=',')
with open('./data/ilsvrc12/synset_words.txt', 'r') as f:
	synsets = f.readlines()
print synsets[1]

vis={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for i in range(1000):
	vis[int(clusters[i])].append(synsets[i].split(" ",1)[1].strip('\n'))
for i in range(10):
	print '========== {0} =========='.format(i)
	print vis[i]
