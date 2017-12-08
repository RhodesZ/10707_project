# Sparse coding baseline model
import csv
import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import SparseCoder, DictionaryLearning
from numpy.linalg import norm

from nltk.corpus import wordnet as wn 
import random

def load_word_embeddings(path='glove.6B.50d.txt'):
    return pd.read_csv(path, sep=" ", header=None, index_col=0,
                       quoting=csv.QUOTE_NONE)

def save_dict(model, path="dictionary.p"):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_dict(path="dictionary.p"):
    with open(path, 'rb') as f:
        return pickle.load(f)

def count_meanings(word_list):
	num_list = []
	count1 = 0
	for word in word_list:
		try:
			syns = wn.synsets(word)
			count = 0
			for syn in syns:
				if syn.lemmas()[0].name() == word:
					count += 1
			num_list.append(count)
		except:
			count1 += 1
			num_list.append(0)
	return num_list


def gold_similarity(word1,word2):
	a_syns = wn.synsets(word1)
	b_syns = wn.synsets(word2)

	sim = 0
	for a in a_syns:
		for b in b_syns:
			temp =  ((a).wup_similarity(b))
			if temp != None:
				sim = max(sim,float((a).wup_similarity(b)))
	return sim


def max_similarity(vec1,vec2,Bases):
	sim = []
	for i in range(len(vec1)):
		if vec1[i] == 0:
			continue
		for j in range(len(vec2)):
			if vec2[j] == 0:
				continue
			base1 = Bases[i,:]
			base2 = Bases[j,:]
			sim.append(vec1[i] * vec2[j]* base1.dot(base2) / (norm(base1) * norm(base2)))

	sim = np.max(sim)
	return sim
def origin_similarity(vec1,vec2):
	return vec1.dot(vec2) / (norm(vec1) * norm(vec2))


dataframe = pd.read_table("./glove_10000_n500_k3/selected_df_10k.csv",index_col = 0,header = None,sep = ",")
#dataframe = load_dict("glove_10000_n500_k3/selected_df_10k.p ")
word_list =  dataframe.index
num_list = np.array(count_meanings(word_list))

matrix = dataframe.as_matrix()
dict1 = load_dict("glove_10000_n500_k3/dictionary_10000_n500_k3.p")
coded = dict1.transform(matrix)

print (coded.shape)
print (dict1.components_.shape)

a = np.where(num_list > 0)
word_list_filtered = word_list[a]
coded = coded[a]
matrix = matrix[a]

for i in range(len(coded)):
	coded[i,:] /= norm(coded[i,:])

sim_list1 = []
sim_list2 = []


for k in range(30000):
	i = random.choice(range(len(coded)))
	j = random.choice(range(len(coded)))
	word1 = word_list_filtered[i]
	word2 = word_list_filtered[j]
	sim1 = gold_similarity(word1,word2)
	sim1 = max_similarity(coded[i,:],coded[j,:],dict1.components_)
	#sim2 = origin_similarity(coded[i,:].dot(dict1.components_),coded[j,:].dot(dict1.components_))
	sim2 = origin_similarity(matrix[i,:],matrix[j,:])
	sim_list1.append(sim1)
	sim_list2.append(sim2)

sim_list1 = np.array(sim_list1)
sim_list2 = np.array(sim_list2)
a = np.where(sim_list1 > 0)
sim_list1 = sim_list1[a]
sim_list2 = sim_list2[a]


from scipy.stats import pearsonr,spearmanr
print (spearmanr(sim_list1,sim_list2))
print (pearsonr(sim_list1,sim_list2))


import matplotlib.pyplot as plt 
plt.scatter(sim_list1,sim_list2,alpha = 0.8)
plt.show()
'''
coded = np.abs(coded)
norm =  np.sum(coded,axis = 1)

for i in range(len(coded)):
	coded[i] /= norm[i]

a = np.where(num_list > 0)
num_list = num_list[a]
coded = coded[a]

print (coded.shape)
a = np.sum((coded >= 0.2),axis = 1)
from scipy.stats import pearsonr,spearmanr
print (spearmanr(a,num_list))
print (pearsonr(a,num_list))


coded.sort(axis = 1)
print (coded)


import matplotlib.pyplot as plt 
#plt.scatter(a,num_list,alpha = 0.8)
plt.scatter(coded[:,-1],coded[:,-2],c = np.log(np.log(num_list)),alpha = 0.8,cmap=plt.cm.Blues)
plt.xlabel("Largest Coefficient Value")
plt.ylabel("Second Largest Coefficient Value")
plt.show()
'''
