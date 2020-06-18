"""
Machine learning pipeline based on chapter 2 of Hands-On Machine Learning with Scikit-Learn and Tensorflow
by Aurélien Gero. Code for this chapter (some of which is adapted here) found at 
https://github.com/ageron/handson-ml.

Build and run a machine learning Pipeline object for yeast DM fitness data.

Author: Serena G. Lotreck 
Date: 17 June 2020
"""
import argparse 

import pandas as pd
import numpy as np
from scipy.stats import randint

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.ensemble import RandomForestRegressor

def main(datapath,y_name):
	"""
	Split data, build pipeline, train and test model. 

	parameters: 
		datapath, str: path to datafile
		y_name, str: name of the column to be predicted
	"""
	# make df with data 
	print('===> Loading the data <===')
	data = pd.read_csv(datapath, sep='\t')

	# split testing and training data
	print('===> Splitting test and train data <===')
	train_set, test_set = train_test_split(data, test_size=0.1)
	print(f'Snapshot of training data: {train_set.head()}')

	# split into x and y 
	y_train = train_set[y_name]
	X_train = train_set.drop([y_name], axis=1)
	y_test = test_set[y_name]
	X_test = test_set.drop([y_name], axis=1)
	
	# make preprocessors 
	print('===> Making preprocessors <===')
	numeric_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='median')),
		('scaler', StandardScaler())])
	categorical_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(handle_unknown='ignore'))])

	# get column types 
	print('===> Getting column types <===')
	numeric_features = train_set.select_dtypes(include=['int64', 'float64']).drop([y_name], axis=1).columns
	categorical_features = train_set.select_dtypes(include=['object']).columns

	# make column transformer 
	print('===> Making ColumnTransformer <===')
	preprocessor = ColumnTransformer(
		transformers=[
			('num', numeric_transformer, numeric_features),
			('cat', categorical_transformer, categorical_features)])

	# build overall pipeline 
	print('===> Building overall pipeline <===')
	rf = Pipeline(steps=[('preprocessor', preprocessor), 
				('regressor', RandomForestRegressor())])
	
	# define hyperparameters for search 
	print('===> Defining hyperparameters for random search <===')
	random_grid = {'regressor__n_estimators': randint(low=1, high=200), #why does he use randint?
			'regressor__max_depth': randint(low=1, high=5),
			'regressor__max_features': randint(low=1, high=5),
		}
	print(f'Hyperparameters to search: {random_grid.keys()}')

	# make random search object
	print('===> Making RandomizedSearchCV object <===')
	search = RandomizedSearchCV(rf, random_grid, n_jobs=-1, cv=5)

	# fit the model 
	print('===> Optimizing hyperparameters and fitting the model <===')
	search.fit(X_train, y_train)
	print(f'Best hyperparameters: {search.best_params_}')

	# apply to test set 
	print('===> Applying to test set <===')
	y_pred = search.predict(X_test)
	y_pred_df = pd.DataFrame(y_pred,columns=['y_pred'])
	print(f'Snapshot of predictions: {y_pred_df.head()}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Build and run ML pipeline (regression)')
	parser.add_argument('datapath', help='path to data file')
	parser.add_argument('y_name', help='name of the column to predict')
	args = parser.parse_args()

	main(args.datapath, args.y_name)
