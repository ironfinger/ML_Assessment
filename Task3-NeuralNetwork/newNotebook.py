#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# %%
df = pd.read_csv('../data/HIV.csv')

# %%

df.head()

# %%

# Create labels column:
raw_labels = df['Participant Condition'].to_numpy()

le = preprocessing.LabelEncoder()
le.fit(raw_labels)
labels = le.transform(raw_labels)
df['Labels'] = labels
df.head()

# %%

features = df[['Alpha', 'Beta', 'Lambda', 'Lambda1', 'Lambda2']]
features.head()
# %%

# Create the test set:
train_val_df, test_df = train_test_split(df, test_size=0.1)

# Create the validation and train set:

train_df, validation_df = train_test_split(train_val_df, test_size=0.1)

# %%

