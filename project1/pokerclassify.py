import numpy as np
from sklearn import svm
from sklearn.lda import LDA
import csv
import progressbar

"""
Taken from http://archive.ics.uci.edu/ml/datasets/Poker+Hand
Format is:

1) S1 "Suit of card #1" 
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

2) C1 "Rank of card #1" 
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

3) S2 "Suit of card #2" 
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

4) C2 "Rank of card #2" 
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

5) S3 "Suit of card #3" 
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

6) C3 "Rank of card #3" 
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

7) S4 "Suit of card #4" 
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

8) C4 "Rank of card #4" 
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

9) S5 "Suit of card #5" 
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs} 

10) C5 "Rank of card 5" 
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King) 

11) CLASS "Poker Hand" 
Ordinal (0-9) 

0: Nothing in hand; not a recognized poker hand 
1: One pair; one pair of equal ranks within five cards 
2: Two pairs; two pairs of equal ranks within five cards 
3: Three of a kind; three equal ranks within five cards 
4: Straight; five cards, sequentially ranked with no gaps 
5: Flush; five cards with the same suit 
6: Full house; pair + different rank three of a kind 
7: Four of a kind; four equal ranks within five cards 
8: Straight flush; straight + flush 
9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush 
"""

def readdata(filename):

	with open(filename,'r') as f:
		data_iter = csv.reader(f, delimiter = ',', quotechar = '"')
		data = [data for data in data_iter]
		data_array = np.asarray(data, dtype = int)   

	return data_array
	
	
def covariance(X):
	"""
	Computes the approximate covariance matrix associated with the matrix X. 
	
	Does not assume that the data is centered (i.e. that each column has mean 0).
	
	Parameters
	----------
	X : ndarray
		Input data. Has n_samples rows and n_attributes columns
		
	Returns
	-------
	S : ndarray
		Covariance matrix
	"""
	
	N = X.shape[0]
	Y = X - np.mean(X, 0)
	S = np.dot(Y.T, Y)/float(N-1)
	return S
	
def priors(X,y):
	"""
	Computes the approximation to the prior probabilities of x belonging to each class k
	
	Parameters
	----------
	X : ndarray
		Input data. Has n_samples rows and n_attributes columns
		
	y : ndarray
		Vector of classes associated with each row of X
		
	Returns
	-------
	P : dict
		dictionary of approximate prior probabilities indexed by class. Has length n_classes.
	"""
	n_classes = len(np.unique(y))
	N = X.shape[0]
	P = {}
	for g in np.unique(y):
		Xg = X[y == g, :]
		P[g] = float(Xg.shape[0])/N
	return P

def means(X,y):
	"""
	Computes the approximation to the means of x given that x belongs to a given class k
	
	Parameters
	----------
	X : ndarray
		Input data. Has n_samples rows and n_attributes columns
		
	y : ndarray
		Vector of classes associated with each row of X
		
	Returns
	-------
	M : dict
		dictionary of approximate mean vectors indexed by class. Has length n_classes.
	"""
	n_classes = len(np.unique(y))
	N = X.shape[0]
	M = {}
	for g in np.unique(y):
		Xg = X[y == g,:]
		Ng = Xg.shape[0]
		M[g] = np.sum(Xg, 0)/float(Ng)
	return M
	
def lda(filename,k,l):
	"""
	Computes the coefficients of the LDA estimator which separates classes k and l for the given data (X,y).
	
	We remark that in the notes for the machine learning course at UH, the professor gave the special case of only
	2 classes and wrote w = wk - wl, b = bk - bl. Then instead of taking the argmax_k over all linear discriminant functions delta_k(x)
	we just need to check whether delta_k > delta_l. This arises from taking log(f_k / f_l) where f_k are the conditional probability densities
	associated with class k
	
	Parameters
	----------
	filename : string
		Input data filename. Has n_samples rows and n_attributes + 1 columns
	
	k : int
	l : int
	
	Returns
	-------
	wk : ndarray
		Vector of LDA weights
	wl : ndarray
	
	bk : float
	bl : float
		Constant intercept term

	"""

	Z = readdata(filename)
	X = Z[:,:-1]
	y = Z[:,-1]
	
	Xk = X[y == k, :]
	Xl = X[y == l, :]
	
	Sk = covariance(Xk)
	Sl = covariance(Xl)
	
	N1 = Xk.shape[0]
	N2 = Xl.shape[0]
	N = N1 + N2
	
	S = float(N1)*Sk/N + float(N2)*Sl/N
	M = means(X,y)
	P = priors(X,y)
	
	wk = np.linalg.solve(S, M[k])
	wl = np.linalg.solve(S, M[l])
	
	bk = np.log(P[k]) - 0.5*np.dot(M[k].T, wk)
	bl = np.log(P[l]) - 0.5*np.dot(M[l].T, wl)
	
	return wk, bk, wl, bl
	
def lda_test(filename, wk, bk, wl, bl, k, l):
	"""
	Computes the percent error in prediction when discriminating between class k and class l.
	
	Parameters
	----------
	filename : string
		Input test data file.
		
	wk : ndarray
		Vector of classes associated with each row of X
	bk : float
		Intercept term for kth class predictor function
	wl : ndarray
	bl : float
	k : int
	l : int
	
	Returns
	-------
	accuracy : float
		ratio of correct predictions to total number of data points considered
	"""
	
	Z = readdata(filename)
	X = Z[:,:-1]
	y = Z[:,-1]
	
	Xk = X[y==k, :]
	Xl = X[y==l, :]
	yk = y[y==k]
	yl = y[y==l]
	
	N = len(yk) + len(yl)
	
	Ek = np.concatenate((np.dot(Xk, wk) + bk, np.dot(Xl, wk) + bk))
	El = np.concatenate((np.dot(Xk, wl) + bl, np.dot(Xl, wl) + bl))
	y_test = (Ek > El)*k + (Ek <= El)*l
	
	accuracy = np.sum(y_test == np.concatenate((yk,yl)))/float(N)
	return accuracy
	
def sklearn_train(filename, classes, learner = 'svm'):
	Z = readdata(filename)
	X = Z[:,:-1]
	y = Z[:,-1]
	index = (y==-1) #initialize to array of False boolean values
	for j in classes:
		index += (y==j)
	
	X = X[index, :]
	y = y[index]
	
	if learner == 'lda':
		clf = LDA()
	elif learner == 'svm':
		clf = svm.SVC()
		
	clf.fit(X, y)
	return clf
	
def sklearn_test(testfilename, clf, classes):
	Z = readdata(testfilename)
	X = Z[:,:-1]
	y = Z[:,-1]
	index = (y==-1)
	for j in classes:
		index += (y==j)
	
	X = X[index,:]
	y = y[index]
	N = len(y)
	
	ytest = clf.predict(X)
	weighted_accuracy = np.sum(ytest == y)/float(N)
	
	accuracies = np.zeros(len(classes))
	for (j,c) in enumerate(classes):
		Xc = X[y==c,:]
		yctest = clf.predict(Xc)
		accuracies[j] = np.sum(yctest == c)/float(Xc.shape[0])
	 
	return weighted_accuracy, accuracies


def bruteforce_classifier_test():
	testfilename = 'poker-hand-testing.data'
	from pokerhandid import idhand, besthand
	Z = readdata(testfilename)
	X = Z[:,:-1]
	y = Z[:,-1]
	N = X.shape[0]
	print N
	
	accuracy = 0
	for j in xrange(N):
		accuracy += (besthand(idhand(X[j,:])) == y[j])
		progressbar.printProgress(j+1, N, prefix = "pokerclassify.bruteforce_classifier_test: iteration {}".format(j+1), barLength = 40)
	
	accuracy = accuracy*1./float(N)
	return accuracy
	

#def keras_nn_train():
#	from keras.models import Sequential
#	from keras.layers import Dense, Dropout, Activation, Flatten
#	from keras.layers import Convolution2D, MaxPooling2D
#	from keras.utils import np_utils
#	from keras import backend as K
#	filename = 'poker-hand-training_true.data'
#	Z = readdata(filename)
#	X_train = Z[:,:-1]
#	y_train = Z[:,-1]
#	N = X.shape[0]
#	print N

#	model = Sequential()
#	model.add(
	

def main():

	filename = 'poker-hand-training-true.data'
	testfilename = 'poker-hand-testing.data'

 	classes = range(0,10)
	classes = [3,9]
 	k = 3
 	l = 9
 	
	wk, bk, wl, bl = lda(filename, k, l)
	a = lda_test(testfilename, wk, bk, wl, bl, k, l)
	print(a)
	
	clf1 = sklearn_train(filename, classes, 'lda')
	clf2 = sklearn_train(filename, classes, 'svm')
	print(sklearn_test(testfilename, clf1, classes))
	print(sklearn_test(testfilename, clf2, classes))
	
	
if __name__ == "__main__":
	main()
	#print bruteforce_classifier_test()
