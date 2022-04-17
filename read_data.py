from nis import cat
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np

def load_dataset(train_file, test_file, protected_features):

	# load the dataset as a numpy array
	dataframe = read_csv(train_file, header=None, na_values='?')
	# drop rows with missing
	dataframe = dataframe.dropna()
	# split into inputs and outputs
	last_ix = len(dataframe.columns) - 1
	X_train, Y_train = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
	Y_train = LabelEncoder().fit_transform(Y_train)	

	X_proc = X_train[protected_features].copy()
	X_train = X_train.drop(protected_features, axis=1)

	X_train.columns = range(X_train.shape[1])
	X_proc.columns = range(X_proc.shape[1])
	
	# select categorical and numerical features
	cat_ix = X_train.select_dtypes(include=['object', 'bool']).columns
	cat_ix_proc = X_proc.select_dtypes(include=['object', 'bool']).columns
	num_ix = X_train.select_dtypes(include=['int', 'float64']).columns
	num_ix_proc = X_proc.select_dtypes(include=['int64', 'float64']).columns
	# label encode the target variable to have the classes 0 and 1
	
	X_train=X_train.values
	X_proc=X_proc.values
		# load the dataset as a numpy array
	dataframe = read_csv(test_file, header=None, na_values='?')
	# drop rows with missing
	dataframe = dataframe.dropna()
	# split into inputs and outputs
	last_ix = len(dataframe.columns) - 1
	X_test, Y_test = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
	Y_test = LabelEncoder().fit_transform(Y_test)
	
	X_test_proc = X_test[protected_features].copy()
	X_test = X_test.drop(protected_features, axis=1)
	
	X_test_proc.columns = range(X_test_proc.shape[1])
	X_test.columns = range(X_test.shape[1])

	X_test=X_test.values
	X_test_proc = X_test_proc.values
	train_size,test_size = X_train.shape[0], X_test.shape[0]

	X = np.concatenate([X_train, X_test], axis=0)
	X_proc = np.concatenate([X_proc, X_test_proc], axis=0)
	
	
	for ix in cat_ix:
		X[...,ix]=LabelEncoder().fit_transform(X[...,ix])
	ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])

	
	X = ct.fit_transform(X)
	
	for ix in cat_ix_proc:
		X_proc[...,ix]=LabelEncoder().fit_transform(X_proc[...,ix])
	# ct = ColumnTr/ansformer([('c',OneHotEncoder(),cat_ix_proc), ('n',MinMaxScaler(),num_ix_proc)])
	# X_proc = ct.fit_transform(X_proc)

	return X[:train_size, ...],X_proc[:train_size, ...], Y_train, X[-test_size:, ...], X_proc[-test_size:, ...], Y_test

if __name__ == "__main__":
	# define the location of the dataset
	full_path = 'adult.data'
	# load the dataset
	X_train, X_proc_train, Y_train, X_test, X_proc_test, Y_test = load_dataset("adult.data", "adult.test", [7,8,9])
	print(X_train.shape, X_proc_train.shape, Y_train.shape, X_test.shape, X_proc_test.shape, Y_test.shape)


