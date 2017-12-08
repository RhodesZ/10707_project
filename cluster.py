import pandas as pd
import csv
import numpy as np
import seaborn as sns ; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.decomposition import PCA
import sklearn
from matplotlib.colors import ListedColormap

def Build_dict():
	vocab = np.load("glove.6B.50d.txt.vocab.npy")

	print "Building the Dict"
	word2index = {}
	index2word = {}
	for i in xrange(len(vocab)):
		word2index[vocab[i]] = i+1
		index2word[i+1] = vocab[i]
	index2word[0] = "PAD"
	np.save("./word2index.npy",word2index)
	np.save("./index2word.npy",index2word)

def index2sentece(indexs):
	strs = ""
	word2index = np.load("./word2index.npy").item()
	index2word = np.load("./index2word.npy").item()
	condensed_map = np.load('./new_predicted/Ytrain_unique.npy')
	for index in indexs:
		strs+=index2word[index]
		strs+=" "
	print strs
def GetFeatureVec(word):

	word2index = np.load("./word2index.npy").item()
	index2word = np.load("./index2word.npy").item()

	condensed_map = np.load('./new_predicted/Ytrain_unique.npy')
	embedding = pd.read_table("./glove.6B.50d.txt",sep = " ",header = None,index_col = 0,quoting=csv.QUOTE_NONE)
	embedding.index = xrange(len(embedding))
	embedding = np.asarray(embedding)
	new_predicted = np.load("./new_predicted/am_%s_predicted.npy" %(word))
	index = np.random.permutation(xrange(len(new_predicted)))[0:1000]
	new_predicted = new_predicted[index]
	result = []


	'''
	sum1 = np.mean(new_predicted,axis = 0)
	order = np.argsort(sum1)[::-1]
	order = order[0:10]
	result = new_predicted[:,order]
	
	words = []
	for i in order:
		print index2word[condensed_map[i]]
		words.append(index2word[condensed_map[i]])
	np.save("./words.npy",words)
	'''
	
	#print sum1.shape
	words = []
	for i in xrange(len(new_predicted)):
		vec = new_predicted[i,:]
		order = np.argsort(vec)[::-1]
		num  = 3
		top_5 = order[:num]
		origin_top_5 = condensed_map[top_5] - 1
		temp = embedding[origin_top_5]
		for i in xrange(len(temp)):
			result.append(temp[i])
			words.append(index2word[condensed_map[top_5[i]]])
		#feature_vec = temp.reshape(50*num)
		#feature_vec = np.mean(temp,axis = 0)
		#result.append(feature_vec)
	result = np.asarray(result)
	np.save("./words.npy",words)
	print result.shape
	np.save("./feature_vec/am_%s.npy" %(word),result)
	
def Clustering(word):
	colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
	X = np.load("./feature_vec/am_%s.npy" %(word))
	
	#pca = sklearn.manifold.TSNE(n_components=2,n_iter=500, n_iter_without_progress=100)
	
	pca = PCA(n_components=2)
	vis_vec = pca.fit_transform(X)
	x = vis_vec[:,0]
	y = vis_vec[:,1]
	print X.shape
	fig1, axes1 = plt.subplots(2, 2, figsize=(8, 8))

	fpcs = []

	for ncenters, ax in enumerate(axes1.reshape(-1), 2):
		print ncenters
		cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, ncenters, 2, error=0.005, maxiter=2000, init=None)

		# Store fpc values for later
		fpcs.append(fpc)

		# Plot assigned clusters, for each data point in training set
		cluster_membership = np.argmax(u, axis=0)
		for j in range(ncenters):
			ax.plot(x[cluster_membership == j],
					y[cluster_membership == j], '.', color=colors[j])
		words = np.load("./words.npy")
		vis = []
		word_vis = []
		for j in range(ncenters):
			vec = X[cluster_membership == j]
			mean_vec = np.mean(vec,axis = 0)
			value = np.sum((vec - mean_vec) ** 2,axis = 1)
			order = np.argsort(value)
			if len(order) == 0:
				continue
			visword = words[(np.arange(3000)[cluster_membership == j])[order[0]]]
			print visword
			word_vis.append(visword)

			vis.append(pca.transform(mean_vec.reshape(1,-1)))
		
		for j in range(len(vis)):
			ax.text(vis[j][0,0],vis[j][0,1],word_vis[j],fontsize = 14)
		
		ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
		ax.axis('off')
	
	'''
	cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, 2, 2, error=0.005, maxiter=2000, init=None)

	cluster_membership = np.argmax(u, axis=0)
	for j in range(2):
		axes1.plot(x[cluster_membership == j],
				y[cluster_membership == j], '.', color=colors[j])
	
	words = np.load("./words.npy")
	vis = []
	word_vis = []
	for j in range(2):
		vec = X[cluster_membership == j]
		mean_vec = np.mean(vec,axis = 0)
		value = np.sum((vec - mean_vec) ** 2,axis = 1)
		order = np.argsort(value)
		word = words[(np.arange(3000)[cluster_membership == j])[order[0]]]
		print word
		word_vis.append(word)

		vis.append(pca.transform(mean_vec.reshape(1,-1)))

	for j in range(2):
		axes1.text(vis[j][0,0],vis[j][0,1],word_vis[j],fontsize = 14)
	'''
	'''
	for pt in cntr:
		pt_ = pca.transform(pt.reshape((1,-1)))
		print pt_
		axes1.plot(pt_[0,0], pt_[0,1], 'rs')
	'''

	fig1.tight_layout()
	#plt.box()
	#plt.show()
	plt.savefig('./cmeans_%s.pdf' %(word))
	
	'''
	X = pd.DataFrame(X)
	X.index = np.load("./words.npy")
	cmap = sns.color_palette("Reds",n_colors = 50)
	ax = sns.clustermap(X,cmap = ListedColormap(cmap))
	plt.setp(ax.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
	
	plt.show()
	plt.savefig('./hclust_%s.pdf' %(word))
	'''
#Build_dict()
for word in ['tie','star','blue','bank','credit','fly','light','mean']:
	GetFeatureVec(word)
	Clustering(word)

#GetFeatureVec("tie")
#Clustering("tie")
