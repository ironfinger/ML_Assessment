"""
TODO: 
        - [] Import the data
        - [] Split by class
        - [] Box plot (x: Status | y: Alpha)
"""

#%%

# Import libraries:
import pandas as pd # Import pandas for dataset exploration.
import matplotlib.pyplot as plt
from sqlalchemy import column # Import matplotlib for data visualisation.

# Get the dataset:
dataset = pd.read_csv('../data/HIV.csv')

dataset.head()
# %%

# Get the patient df:
patient_df = dataset.loc[dataset['Participant Condition'] == 'Patient']
patient_df.head()

# Get the control df:
control_df = dataset.loc[dataset['Participant Condition'] == 'Control']
control_df.head()

# %%

# Display boxplot by Status on X and Alpha on y:
dataset.boxplot(by='Participant Condition', column='Alpha')


# %%

# Density plot between 

# Get numpy array for beta Patient:
beta_patient = patient_df['Beta'].to_numpy()
print(beta_patient.shape)

# Get numpy array for beta Control:
beta_control = control_df['Beta'].to_numpy()
print(beta_control.shape)
# %%

ax_01 = pd.Series(beta_patient)
ax_02 = pd.Series(beta_control)

density_df = pd.DataFrame({
    'Patient Beta': beta_patient,
    'Control Beta': beta_control
})

density_ax = density_df.plot.kde()
plt.show()

# %%

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(dataset, test_size=0.2)


# %%
train_df.head()

# %%
test_df.head()

#%%

print(dataset['Alpha'].min())

#%%
min_dic = {}
min_dic['Summary'] = 'Min'

for c in dataset.columns:
    min_dic[c] = [dataset[c].min()]

min_df = pd.DataFrame(data=min_dic)
min_df
# %%

class_names = ['Patient', 'Control']

# Two hidden layers each with 500 neurons sigmoid 
# for hidden layers and logistic function for output

import tensorflow as tf
from tensorflow import keras 



