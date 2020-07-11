"""
Script to run RadiusNeighborsClassifier on gene pathway data and calculate performance measures.

Author: Serena G. Lotreck 
Date: 07/09/2020
"""
import argparse 

import pandas as pd
import numpy as np 
from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import f1_score 

def split_train_test(data, train, test):
	"""
	Splits test and train data based on supplied indices. 

	Parameters:
		data, df: feature matrix with labels 
		train, df: column with ID's of training instances 
		test, df: column with ID's of test instances  

	returns: (X_train, y_train, X_test, y_test)
	"""

	# Separate test and train data 
	train_inst = data[data.index.isin(train['train_ID'])]
	test_inst = data[data.index.isin(test['test_ID'])]
	
	X_cols = [c for c in data.columns if c.lower()[:3] != 'pwy']
	y_cols = [c for c in data.columns if c.lower()[:3] == 'pwy']
	
	X_train = train_inst[X_cols] 
	y_train = train_inst[y_cols]

	X_test = test_inst[X_cols]
	y_test = test_inst[y_cols]

	print(f'Snapshot of X_train:\n{X_train.head()}\nSnapshot of y_train:\n{y_train.head()}')

	return (X_train, y_train, X_test, y_test)


def make_pipeline(X_train):
	"""
	Make Pipeline object.

	Parameters:
		X_train, df: Features for training data

	returns: Pipeline object 
	"""
	# Make preprocessors
	numeric_transformer = Pipeline(steps=[
					('imputer', SimpleImputer(strategy='median')),
					('scaler', StandardScaler())])
	categorical_transformer = Pipeline(steps=[
					('imputer', SimpleImputer(strategy='most_frequent')),
					('onehot', OneHotEncoder(handle_unknown='ignore'))])

	# Get column types 
	numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
	categorical_features = X_train.select_dtypes(include=['object']).columns

	# Make column transformer
	preprocessor = ColumnTransformer(transformers=[
					('num', numeric_transformer, numeric_features),
					('cat', categorical_transformer, categorical_features)])

	# Make overall pipeline
	rNeighbors = Pipeline(steps=[
				('preprocessor', preprocessor),
				('classifier', RadiusNeighborsClassifier(outlier_label='most_frequent'))])

	return rNeighbors
	

def tune_hyperparameters(pipe, X_train, y_train):
	"""
	Use randomized search to define best hyperparameters. 

	Parameters:
		pipe, Pipeline object: full pipeline for the model
		X_train, df: feature matrix without labels for training data
		y_train, df: labels for training data

	returns: search, a RandomizedSearchCV object 
	"""
	# Define hyperparameters for search
	random_grid = {'classifier__radius': randint(low=1, high=100),
			'classifier__weights': ['uniform', 'distance']}

	# Make RandomSearchCV object
	search = RandomizedSearchCV(pipe, random_grid, n_iter=20, n_jobs=1, cv=10, random_state=40)

	# Fit search object 
	search.fit(X_train, y_train)
	print(f'Best hyperparameters are: {search.best_params_}')

	return search 


def main(data, train, test):
	"""
	Function to build, run, and evaluate model. 

	Parameters:
		data, df: feature matrix with labels 
		train, df: column with ID's of training instances 
		test, df: column with ID's of test instances  
	"""
	# Split training and testing according to pre-defined indices 
	X_train, y_train, X_test, y_test = split_train_test(data, train, test)
	
	# Make Pipeline object 
	pipe = make_pipeline(X_train)

	# Hyperparameter search (this is also model training)
	search = tune_hyperparameters(pipe, X_train, y_train)

	# Apply to test set 
	y_pred = search.predict(X_test)
	
	# Get F1 score
	avg_score = f1_score(y_test, y_pred, average='weighted')
	ind_score = f1_score(y_test, y_pred, average=None)
	print(f'Model performance:\nAverage F1: {avg_score}\nIndividual scores: {ind_score}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Build, run and evaluate ML pipeline for pathway data')
	parser.add_argument('data', help='path to feature matrix')
	parser.add_argument('training', help='path to train set IDs')
	parser.add_argument('test', help='path to test set IDs')

	args = parser.parse_args()

	data = pd.read_csv(args.data, sep='\t', index_col=0)
	training = pd.read_csv(args.training, sep='\t', names=['train_ID'])
	test = pd.read_csv(args.test, sep='\t', names=['test_ID'])
	
	main(data, training, test)
