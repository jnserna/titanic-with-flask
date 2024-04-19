'''
Author ~ Juanes Serna
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_original = pd.read_csv("titanic2.csv")

# Setting up the new data types
dtypes_col       = data_original.columns
dtypes_type_old  = data_original.dtypes
dtypes_type      = ['int16', 'bool','category','object','category','float32','int8','int8','object','float32','object','category']
optimized_dtypes = dict(zip(dtypes_col, dtypes_type))

# Reading the entire data, but now with optimized columns
data = pd.read_csv("titanic2.csv",dtype=optimized_dtypes)

from random import choices

x = data["Age"].dropna()
hist, bins = np.histogram(x,bins=30)
# Finding the probability of Age
bincenters = 0.5*(bins[1:]+bins[:-1])
probabilities = hist/hist.sum()

# Creating random numbers from existing age distribution
for item in data['Age']:
    data["Age_rand"] = data["Age"].apply(lambda v: np.random.choice(bincenters, p=probabilities))
    Age_null_list   = data[data["Age"].isnull()].index

    # Filling...   
    data.loc[Age_null_list,"Age"] = data.loc[Age_null_list,"Age_rand"]
    
data = data.drop(columns = ["Age_rand"])

data["Embarked"] = data["Embarked"].fillna("S")

# OneHot encoder for Sex
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, dtype=int)
data[['female', 'male']] = encoder.fit_transform(data[['Sex']])
data = data.drop(['Sex'], axis=1)

# label encoding - variable Embarked

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data["Embarked"]=le.fit_transform(data["Embarked"])

# label encoding - variable Pclass

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data["Pclass"]=le.fit_transform(data["Pclass"])
# Ojo que con el encoder la PClass pasa de 3,1,2 a 2,0,1. Esto es importante para los datos del html.

# 1. Creating features and targets
# 1. Definición de variables predictoras y variable a predecir

X4 = data[['Pclass','Age','female','male']] # Dropping Cabin and Passenger Id, and Embarked (beacuse its low feature importance)
y4 = data['Survived']

# 2. Splitting dataset into training data and validation data 
# 2. Separación de los datos en datos para entrenamiento y datos para validación

from sklearn.model_selection import train_test_split
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.3, random_state=42)

# 3. Creating instance of the model
# 3. Instanciar el modelo

from sklearn.tree import DecisionTreeClassifier
profundidad = 6
tree4 = DecisionTreeClassifier(max_depth=profundidad)

# 4. Training the model
# 4. Entrenar el modelo

tree4.fit(X4_train, y4_train)

#joblib

from joblib import dump

dump(tree4, 'tree4.joblib')