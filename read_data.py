from nis import cat
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np

def load_dataset(train_file, test_file):

	# load the dataset as a numpy array
	dataframe = read_csv(train_file, header=None, na_values='?')
	# drop rows with missing
	dataframe = dataframe.dropna()
	# split into inputs and outputs
	last_ix = len(dataframe.columns) - 1
	X_train, Y_train = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
	Y_train = LabelEncoder().fit_transform(Y_train)
	# select categorical and numerical features
	cat_ix = X_train.select_dtypes(include=['object', 'bool']).columns
	num_ix = X_train.select_dtypes(include=['int64', 'float64']).columns
	# label encode the target variable to have the classes 0 and 1
	X_train=X_train.values

	# load the dataset as a numpy array
	dataframe = read_csv(test_file, header=None, na_values='?')
	# drop rows with missing
	dataframe = dataframe.dropna()
	# split into inputs and outputs
	last_ix = len(dataframe.columns) - 1
	X_test, Y_test = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
	Y_test = LabelEncoder().fit_transform(Y_test)
	# select categorical and numerical features
	cat_ix = X_test.select_dtypes(include=['object', 'bool']).columns
	num_ix = X_test.select_dtypes(include=['int64', 'float64']).columns
	# label encode the target variable to have the classes 0 and 1
	X_test=X_test.values

	train_size,test_size = X_train.shape[0], X_test.shape[0]

	X = np.concatenate([X_train, X_test], axis=0)

	for ix in cat_ix:
		X[...,ix]=LabelEncoder().fit_transform(X[...,ix])
	ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])
	X = ct.fit_transform(X)
	return X[:train_size, ...], Y_train, X[-test_size:, ...], Y_test

if __name__ == "__main__":
	# define the location of the dataset
	full_path = 'adult.data'
	# load the dataset
	X_train, Y_train, X_test, Y_test = load_dataset("adult.data", "adult.test")

	print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


