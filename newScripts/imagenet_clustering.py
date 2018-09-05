import numpy as np
import sklearn.cluster as cls

def save_mat(filename,mat):
	with open(filename,'w') as f:
		np.savetxt(f,mat,delimiter=',')

def load_mat(filename):
	return np.genfromtxt(filename, delimiter=',')

def gen_conf_mat(num_label):
	label = load_mat('label.csv')
	prediction = load_mat('prediction.csv')
	pred_prob = np.zeros((num_label, num_label))
	freq_label = np.zeros(num_label)
	for label_ind, label_value in enumerate(label):
		label_int = int(label_value)
		pred_prob[label_int,:] += prediction[label_ind,:]
		freq_label[label_int] += 1

	freq_mat = np.zeros((num_label, num_label))
	for i in range(num_label):
		for j in range(num_label):
			freq_mat[i,j] = freq_label[i]+freq_label[j]
	
	#for uniq_label in range(num_label):
	#	avg_prob[uniq_label,:] /= freq_label[uniq_label]
	confusion_mat = 0.5*(pred_prob + np.transpose(pred_prob))
	norm_conf_mat = np.divide(confusion_mat, freq_mat)

	save_mat('conf_mat.csv',confusion_mat)
	save_mat('norm_conf_mat.csv',norm_conf_mat)

def use_kmeans(mat, n_cluster):
	kmeans_clusters = cls.KMeans(n_clusters=n_cluster).fit(mat)
	kmeans_hist, kmeans_bin_edges = np.histogram(kmeans_clusters.labels_, bins=np.arange(n_cluster+1))
	print 'kmeans clustering:', kmeans_clusters.labels_
	print kmeans_hist
	return kmeans_clusters.labels_

def use_specCluster(mat, n_cluster):
	clusters = cls.spectral_clustering(affinity=mat, n_clusters=n_cluster)
	hist, bin_edges = np.histogram(clusters, bins=np.arange(n_cluster+1))
	print 'spectral clustering:', clusters
	print hist, bin_edges
	return clusters, hist

#damping = 0.99282 -> 9 clusters; 0.9928 -> 10 clusters; 0.9929 -> 8 clusters
def use_af(mat, n_cluster):
	clusters = cls.AffinityPropagation(damping=0.99282, affinity='precomputed').fit(mat)
	n_cluster = max(clusters.labels_)+1
	hist, bin_edges = np.histogram(clusters.labels_, bins=np.arange(n_cluster+1))
	print 'Affinity Propagation clustering:', clusters.labels_
	print hist
	return clusters.labels_


def use_meanShift(mat, n_cluster):
	clusters = cls.MeanShift().fit(mat)
	n_cluster = max(clusters.labels_)+1
	hist, bin_edges = np.histogram(clusters.labels_, bins=np.arange(n_cluster+1))
	print 'Mean Shift clustering:', clusters.labels_
	print hist
	return clusters.labels_

def use_ac(mat, n_cluster):
	clusters = cls.AgglomerativeClustering(n_clusters=n_cluster).fit(mat)
	hist, bin_edges = np.histogram(clusters.labels_, bins=np.arange(n_cluster+1))
	print 'Agglomerative clustering:', clusters.labels_
	print hist
	return clusters.labels_

def use_birch(mat, n_cluster):
	clusters = cls.Birch(threshold=0.0005,n_clusters=n_cluster).fit(mat)
	hist, bin_edges = np.histogram(clusters.labels_, bins=np.arange(n_cluster+1))
	print 'Birch clustering:', clusters.labels_
	print hist
	return clusters.labels_

num_label = 1000
n_cluster = 9

#gen_conf_mat(num_label)
norm_conf_mat = load_mat('norm_conf_mat.csv')

#clusters, hist = use_specCluster(norm_conf_mat, n_cluster)
#save_mat('{0}cluster_oneLevel.csv'.format(n_cluster),clusters)

### two level spectral clustering #########
#clusters, hist = use_specCluster(norm_conf_mat, n_cluster)
#hist_ind = np.where(hist>500)
#print hist_ind
#remain = np.squeeze(clusters==hist_ind)
#remain_ind = np.squeeze(np.where(clusters == np.squeeze(hist_ind)))
#new_conf_mat = norm_conf_mat[remain,:]
#new_conf_mat = new_conf_mat[:,remain]
#print new_conf_mat.shape
#
#new_clusters, new_hist = use_specCluster(new_conf_mat, 5)
#for i in range(len(remain_ind)):
#	clusters[remain_ind[i]] = n_cluster + new_clusters[i]
#
#print clusters
#save_mat('9cluster.csv',clusters)

# highly unbalanced
#kmeans_clusters = use_kmeans(norm_conf_mat, n_cluster)
#save_mat('{0}cluster_kmeans.csv'.format(n_cluster),kmeans_clusters)

#clusters = use_af(norm_conf_mat, n_cluster)
#save_mat('{0}cluster_af.csv'.format(n_cluster),clusters)
#clusters = use_ac(norm_conf_mat, n_cluster)
#save_mat('{0}cluster_ac.csv'.format(n_cluster),clusters)
clusters = use_birch(norm_conf_mat, n_cluster)
save_mat('{0}cluster_birch.csv'.format(n_cluster),clusters)
