from nis import cat
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np

df = pd.read_csv('/Users/faisalhamman/Desktop/DeepLearning/Unbiasing-ML-Models-master/adult-all.csv', header=None, na_values='?')
df.columns =['age', 'workclass', 'fnlwgt', 'education', 
                    'education_num', 'marital_status', 'occupation', 
                    'relationship', 'race', 'sex', 'capital_gain', 
                   'capital_loss', 'hours_per_week', 'native_country', 'salary']

listA = ['Wife','Husband']

def label_race (row):
   if row['relationship'] in listA:
      return 'Married'
   else :
      return 'Unmarried'
    
df['relationship'] = df.apply (lambda row: label_race(row), axis=1)


df.to_csv('/Users/faisalhamman/Desktop/DeepLearning/Unbiasing-ML-Models-master/adult_marry.csv',header=False)
