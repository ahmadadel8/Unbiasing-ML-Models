from nis import cat
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np

df = pd.read_csv('adult.data', header=None, na_values='?',)
df.columns =['age', 'workclass', 'fnlwgt', 'education', 
                    'education_num', 'marital_status', 'occupation', 
                    'relationship', 'race', 'sex', 'capital_gain', 
                   'capital_loss', 'hours_per_week', 'native_country', 'salary']

df = df.drop('relationship', axis=1)


df.to_csv('adult_wo_relationship.data',header=False, index=False)
