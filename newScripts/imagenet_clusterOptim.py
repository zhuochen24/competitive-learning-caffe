import cvxpy as cvx
import numpy as np

n_cluster = 9
error_cons=0.23
ifInt = 0
obj = 'max_sq'
singleCluster = 0
mapping_file='./clustering/{0}cluster_oneLevel.csv'.format(n_cluster)

### map both label and prediction to clusters ######
def map_to_cluster(pred, label):
        mapping = np.genfromtxt(mapping_file,delimiter=',')
        new_label = np.copy(label)
        for ind, value in enumerate(label):
                new_label[ind] = mapping[int(value)]
        new_pred = np.copy(pred)
        for ind, value in enumerate(pred):
                new_pred[ind] = mapping[int(value)]
        return new_pred, new_label


def save_mat(filename,mat):
        with open(filename,'w') as f:
                np.savetxt(f,mat,delimiter=',')

def analyze_solution(mat):
	print 'Sum of each row before binarization'
	print np.sum(mat,axis=1)
	mat[mat<=0.5] = 0
	mat[mat>=0.5] = 1
	print 'number of classes per cluster:'
	print np.sum(mat,axis=1)
	print 'total number of classes:'
	print np.sum(mat)
	print 'Its percentage'
	print np.sum(mat,axis=1)/1000.0

	print 'number of clusters per class:'
	print np.sum(mat,axis=0)
	print 'Its percentage'
	print np.sum(mat,axis=0)/9.0

def optimize(n_cluster, error_cons, pred_file, label_file, solution_file , ifInt=0, obj='balance', singleCluster = False):
	# Problem data.
	n_class = 1000

	#mapping = np.genfromtxt('9cluster.csv',delimiter=',')
	#ori_label = np.genfromtxt('label_squeezenet_9cluster.csv', delimiter=',')
	ori_pred = np.genfromtxt(pred_file, delimiter=',')
	ori_top1_pred = ori_pred.argmax(1)
	class_label = np.genfromtxt(label_file, delimiter=',')
	ori_top1_pred, class_label = map_to_cluster(ori_top1_pred, class_label)

	# Construct the problem.
	if ifInt:
		P = cvx.Bool(n_cluster, n_class)
		constraints = []
		use_solver = cvx.GUROBI
	else:
		P = cvx.Variable(n_cluster, n_class)
		constraints = [0 <= P, P <= 1]
		use_solver = cvx.ECOS

	if singleCluster:
		constraints.append(cvx.sum_entries(P,axis=0) <= 1)

	# Objective function
	if obj == 'sum':
		obj_func = cvx.sum_entries(P)
	elif obj == 'sum_sq':
		obj_func = cvx.sum_squares(cvx.sum_entries(P,axis=1))
	elif obj == 'max':
		obj_func = cvx.max_entries(cvx.sum_entries(P,axis=1))
	elif obj == 'max_sq':
		obj_func = cvx.max_entries(cvx.power(cvx.sum_entries(P,axis=1),2))
	else:
		obj_func = cvx.sum_squares(cvx.sum_entries(P, axis=1) - n_class/float(n_cluster))

	objective = cvx.Minimize(obj_func)

	# Error rate constraint
	error_constraint = 0
	for cluster, label in zip(ori_top1_pred, class_label):
		error_constraint += P[cluster, label]
	error_constraint = 1.0 - 1.0/len(class_label) * error_constraint
	constraints.append(error_constraint <= error_cons)


	# The optimal objective is returned by prob.solve().
	prob = cvx.Problem(objective, constraints)
	result = prob.solve(solver = use_solver)

	save_mat(solution_file,P.value)

	print 'status:', prob.status
	print 'optimal value: ', prob.value
	print 'Optimal P matrix:', P.value
	print 'Error rate constraint dual value: ', constraints[-1].dual_value
	print 'Error rate constraint {0}:{1} <= {2}'.format(constraints[-1].value, error_constraint.value,error_cons)


for obj in ['balance', 'sum','sum_sq', 'max']:
	print '===================== start new batch: {0}============================'.format(obj)
	for error_cons in [0.15, 0.10, 0.5]:
		for ifInt in [0]:
			print '======================================================='
			solution_file = 'ifInt{2}_sol_{0}cluster_{1}_obj{3}_singleCluster{4}_rightLabel.csv'.format(n_cluster,error_cons, ifInt,obj,singleCluster)
			print solution_file
			optimize(n_cluster, error_cons, \
				 #pred_file='prediction_squeezenet_9cluster.csv',\
				 #label_file='label_squeezenet.csv', \
				 pred_file = 'prediction_squeezenet.csv',\
				 label_file='label_squeezenet.csv',\
				 solution_file = solution_file,\
				 ifInt = ifInt, \
				 obj = obj,
				 singleCluster = singleCluster)
			solution = np.genfromtxt(solution_file,delimiter=',')
			analyze_solution(solution)
