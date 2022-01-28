import numpy as np
import os
import argparse
from scipy.sparse import csr_matrix, coo_matrix, triu
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import random
import pickle

def load_data(path):
	'''
	读xxx2id
	返回下标为id，值为xxx的数组
	'''
	data_raw = []
	data = open(path, 'r')
	num_data = int(data.readline().strip())
	for i in range(num_data):
		line = data.readline().strip().split()
		data_raw.append((int(line[1]), line[0]))
	data_raw.sort()

	data_info = []
	for pair in data_raw:
		data_info.append(pair[1])
	data.close()

	return num_data, data_info

def generate_train_tuple(train, size_train):
	train_tuple = []

	for _ in range(size_train):
		line = train.readline()
		line = line.replace('\t', ' ')
		triplet_str = line.strip().split(' ')
		train_tuple.append(tuple(map(int, triplet_str)))

	return train_tuple

def generate_hypergraph(train_tuple):	
	dict_head = dict()
	dict_tail = dict()

	for (e1, e2, r) in train_tuple:
		# Gather all hyperedges sharing same head
		if e1 not in dict_head:
			dict_head[e1] = dict()

		if r not in dict_head[e1]:
			dict_head[e1][r] = []

		dict_head[e1][r].append(e2)

		# Gather all hyperedges sharing same tail
		if e2 not in dict_tail:
			dict_tail[e2] = dict()

		if r not in dict_tail[e2]:
			dict_tail[e2][r] = []

		dict_tail[e2][r].append(e1)

	# Updating hyperedges
	hyperedges = []

	for entity in dict_head:
		for rel in dict_head[entity]:
			if len(dict_head[entity][rel]) > 1:
				hyperedges.append(tuple(dict_head[entity][rel]))

	for entity in dict_tail:
		for rel in dict_tail[entity]:
			if len(dict_tail[entity][rel]) > 1:
				hyperedges.append(tuple(dict_tail[entity][rel]))

	return hyperedges

def A_hat_csr(hyperedges, num_ent, weight, normalize):
	indptr = [0]
	indices = []
	data = []

	for i in range(len(hyperedges)):
		for node in hyperedges[i]:
			indices.append(node)
			if normalize == True:
				data.append(weight[i] ** 2)
			else:
				data.append(weight[i])
		indptr.append(len(indices))

	FDAT = csr_matrix((data, indices, indptr), dtype = float, shape = (len(hyperedges), num_ent))
	# print("FDAT:", FDAT.shape)
	A = csr_matrix((np.ones(len(indices)), indices, indptr), dtype = float, shape = (len(hyperedges), num_ent)).transpose()
	# print("A:", A.shape)

	A.indptr = np.array(A.indptr, copy = False, dtype = np.int64)
	A.indices = np.array(A.indices, copy = False, dtype = np.int64)

	FDAT.indptr = np.array(FDAT.indptr, copy = False, dtype = np.int64)
	FDAT.indices = np.array(FDAT.indices, copy = False, dtype = np.int64)

	return A @ FDAT

#---main function---

random.seed(10000)

parser = argparse.ArgumentParser()
parser.add_argument("data", help = "folder name that contains data", default = None)
parser.add_argument("density", help = "density of clustered graph")
args = parser.parse_args()

data = args.data
density = float(args.density)

path_data = './benchmarks/' + data + '/'
path_save = './result_square/' + str(density) + '/' + data + '_metad_embedding/'

if not os.path.exists(path_save):
	os.makedirs(path_save)

num_ent, entity_info = load_data(path_data + 'entity2id.txt')
num_rel, relation_info = load_data(path_data + 'relation2id.txt')

train = open(path_data + 'train2id.txt', 'r')
size_train = int(train.readline().strip())
train_tuple = generate_train_tuple(train, size_train)

train.close()
'''
hyperedges = generate_hypergraph(train_tuple)
num_edges = len(hyperedges)
deg_e_inv = []

for edge in hyperedges:
	deg_e_inv.append(1 / len(edge))

A_hat = A_hat_csr(hyperedges, num_ent, deg_e_inv, True)

A_hat_sum = A_hat.sum(axis = 0)
A_hat_sum[A_hat_sum == 0] = 1
A_hat_normalized = A_hat / A_hat_sum
A_hat_normalized = A_hat_normalized + A_hat_normalized.transpose()

print("A_hat generated.")
'''
#改一下这句
relation_embeddings = []
entity_embeddings = []
SAVE_PATH='./embedding/'
f1 = open('./embedding/'+data+'/relation_embeddings_origin', 'rb')
f2 = open('./embedding/'+data+'/entity_embeddings_origin', 'rb')
relation_embeddings = pickle.load(f1)
entity_embeddings = pickle.load(f2)
print(relation_embeddings.shape, entity_embeddings.shape)

'''
model = AgglomerativeClustering(n_clusters=int(density * num_ent), linkage="average", affinity="precomputed")
model.fit(- A_hat_normalized)
labels = model.labels_
'''
#加速 n_init
kmeans = KMeans(n_clusters=int(density * num_ent), random_state=0,n_init=1,n_jobs=8).fit(entity_embeddings)
labels = kmeans.labels_
print("Clustering done.")

num_clust = len(set(labels))
clust_size = np.zeros(num_clust)
for i in labels:
	clust_size[i] += 1

fclust = open(path_save + 'labels_' + data + '_' + str(density) +'.txt', 'w')
for i in range(num_ent):
	fclust.write("%s\n" % labels[i])
fclust.close()
#构建超图
train_meta = set()
A_meta = dict()
train_ent = set()
train_rel = set()

for (e1, e2, r) in train_tuple:
	if labels[e1] != labels[e2]:
		train_meta.add((labels[e1], labels[e2], r))
		if (labels[e1], labels[e2], r) not in A_meta:
			A_meta[(labels[e1], labels[e2], r)] = 0
		A_meta[(labels[e1], labels[e2], r)] += 1
		train_ent.add(labels[e1])
		train_ent.add(labels[e2])
		train_rel.add(r)

train_meta = list(train_meta)
print(len(train_meta))

train_filter = []
for (e1, e2, r) in train_meta:
	prob = A_meta[(e1, e2, r)] / (clust_size[e1] * clust_size[e2])
	if random.random() < prob:
		train_filter.append((e1, e2, r))
#---按格式输出文件---
entity_meta = open(path_save + 'entity2id.txt', 'w')
entity_meta.write("%s\n" % (num_clust))
for i in range(num_clust):
	entity_meta.write("%s %s\n" % (i, i))
entity_meta.close()

f_meta = open(path_save + 'train2id.txt', 'w')
f_meta.write("%s\n" % (len(train_filter)))
for (e1, e2, r) in train_filter:
	f_meta.write("%s %s %s\n" % (e1, e2, r))
f_meta.close()
train_filter = set(train_filter)

relation_meta = open(path_save + 'relation2id.txt', 'w')
relation = open(path_data + 'relation2id.txt', 'r')
for line in relation.readlines():
	relation_meta.write(line)
relation_meta.close()
relation.close()

valid = open(path_data + 'valid2id.txt', 'r')
size_valid = int(valid.readline().strip())

valid_tuple = generate_train_tuple(valid, size_valid)
valid.close()

valid_meta = set()
V_meta = dict()

for (e1, e2, r) in valid_tuple:
	if labels[e1] != labels[e2]:
		valid_meta.add((labels[e1], labels[e2], r))
		if (labels[e1], labels[e2], r) not in V_meta:
			V_meta[(labels[e1], labels[e2], r)] = 0
		V_meta[(labels[e1], labels[e2], r)] += 1

valid_meta = list(valid_meta)
print(len(valid_meta))

valid_filter = []
for (e1, e2, r) in valid_meta:
	if (e1, e2, r) not in train_filter:
		prob = V_meta[(e1, e2, r)] / (clust_size[e1] * clust_size[e2])
		if random.random() < prob:
			valid_filter.append((e1, e2, r))
print(len(valid_filter))

f_meta = open(path_save + 'valid2id.txt', 'w')
f_meta.write("%s\n" % (len(valid_filter)))
for (e1, e2, r) in valid_filter:
	f_meta.write("%s %s %s\n" % (e1, e2, r))
f_meta.close()
valid_filter = set(valid_filter)

test = open(path_data + 'test2id.txt', 'r')
size_test = int(test.readline().strip())

test_tuple = generate_train_tuple(test, size_test)
test.close()

test_meta = set()
T_meta = dict()

for (e1, e2, r) in test_tuple:
	if labels[e1] != labels[e2]:
		test_meta.add((labels[e1], labels[e2], r))
		if (labels[e1], labels[e2], r) not in T_meta:
			T_meta[(labels[e1], labels[e2], r)] = 0
		T_meta[(labels[e1], labels[e2], r)] += 1

test_meta = list(test_meta)
print(len(test_meta))

test_filter = []
for (e1, e2, r) in test_meta:
	if (e1, e2, r) not in train_filter and (e1, e2, r) not in valid_filter:
		prob = T_meta[(e1, e2, r)] / (clust_size[e1] * clust_size[e2])
		if random.random() < prob:
			test_filter.append((e1, e2, r))
print(len(test_filter))

f_meta = open(path_save + 'test2id.txt', 'w')
f_meta.write("%s\n" % (len(test_filter)))
for (e1, e2, r) in test_filter:
	f_meta.write("%s %s %s\n" % (e1, e2, r))
f_meta.close()