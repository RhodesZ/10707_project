import pandas as pd 
import numpy as np

vocab = np.load("glove.6B.50d.txt.vocab.npy")

print "Building the Dict"
vocab_dict = {}
for i in xrange(len(vocab)):
	vocab_dict[vocab[i]] = i+1

np.save("./vocab_dict.npy",vocab_dict)

Ambiguous = ["star","bank","tie","fly","mean","credit","blue","Light"]
Specific = ["sleepless","imperfect","abrogate","ceramics","spoken","Connecticut"]

X= np.load("X.npy")
Y = np.load("Y.npy")

def index2sentece(indexs):
	strs = ""
	word2index = np.load("./word2index.npy").item()
	index2word = np.load("./index2word.npy").item()
	
	index2word[0] = "PAD"
	condensed_map = np.load('./predicted/Ytrain_unique.npy')
	for index in indexs:
		strs+=index2word[index]
		strs+=" "
	print strs

for word in Ambiguous:
	y = vocab_dict[word.lower()]
	index = (Y == y)
	sentence = X[index]
	print len(sentence)
	sentence0 = sentence[0]
	index2sentece(sentence0)
	np.save("am_"+word+".npy",sentence)

for word in Specific:
	y = vocab_dict[word.lower()]
	index = (Y == y)
	sentence = X[index]
	print len(sentence)
	#print sentence
	np.save("sp_"+word+".npy",sentence)

print len(sentence[0])