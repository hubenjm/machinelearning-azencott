import numpy as np
from sklearn import svm
import csv

def readdata(filename):

	with open(filename,'r') as f:
		data_iter = csv.reader(f, delimiter = ',', quotechar = '"')
		data = [data for data in data_iter]
		data_array = np.asarray(data, dtype = float)   

	return data_array
	
def svmfit_poker(filename):
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
	
	X = readdata(filename)
	y = X[:,9]
	x = X[:,0:9]
	lin_clf = svm.LinearSVC()
	lin_clf.fit(x,y)
	coeff = lin_clf.coef_
	intercept = lin_clf.coef_
	
	print(coeff)
	
	
def main():

#	filename = 'poker-hand-training-true.data'
#	X = readdata(filename)
#	print(X[:,0])

	filename = 'poker-hand-training-true.data'
	svmfit_poker(filename)
	
if __name__ == "__main__":
	main()
