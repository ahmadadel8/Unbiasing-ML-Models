from nis import cat
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np

def load_dataset(full_path):
	# load the dataset as a numpy array
	dataframe = read_csv(full_path, header=None, na_values='?')
	# drop rows with missing
	dataframe = dataframe.dropna()
	# split into inputs and outputs
	last_ix = len(dataframe.columns) - 1
	X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
	# select categorical and numerical features
	cat_ix = X.select_dtypes(include=['object', 'bool']).columns
	num_ix = X.select_dtypes(include=['int64', 'float64']).columns
	# label encode the target variable to have the classes 0 and 1
	X=X.values
	for ix in cat_ix:
		X[...,ix]=LabelEncoder().fit_transform(X[...,ix])
	ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])
	X = ct.fit_transform(X)
	y = LabelEncoder().fit_transform(y)
	return X, y

def shuffle_and_split(X,Y, split):
	idx = np.arange(X.shape[0])
	np.random.shuffle(idx)
	X = X[idx,...]
	Y = Y[idx,...]
	train_split = np.ceil(X.shape[0]*split).astype(int)
	test_split = X.shape[0]-train_split
	X_train,Y_train, X_test, Y_test = X[0:train_split], Y[0:train_split],X[test_split:], Y[test_split:]
	return X_train,Y_train, X_test, Y_test


if __name__ == "__main__":
	# define the location of the dataset
	full_path = 'adult-all.csv'
	# load the dataset
	X, y = load_dataset(full_path)
	# print(X[0])



