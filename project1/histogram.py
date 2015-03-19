import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

def histogram(x, bins):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	counts, bins, patches = ax.hist(x, bins = bins, color='blue', normed=True, alpha=0.8)
	ax.set_xticks(bins)
	plt.show()
	
def readdata(filename):

	with open(filename,'r') as f:
		data_iter = csv.reader(f, delimiter = ',', quotechar = '"')
		data = [data for data in data_iter]
		data_array = np.asarray(data, dtype = int)   

	return data_array
	
def main():
	filename = sys.argv[1]
	attribute = int(sys.argv[2])
	
	X = readdata(filename)
	x = X[:,attribute-1]
	
	if attribute == 11:
		bins = np.arange(11)
	else:
		bins = np.linspace(1,np.max(x) + 1, np.max(x)+1)
		
	histogram(x, bins)
	
if __name__ == "__main__":
	main()
	
