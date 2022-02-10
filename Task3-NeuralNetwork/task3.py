import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

"""
TODO:
- Mean Values
- Standard Deviations
- Min/max values
"""

def display_boxplot(data):
    data.boxplot(by='Participant Condition', column='Alpha')
    plt.show()
    
def display_density(df_01, df_02):
    axis_1 = pd.Series(df_01['Beta'].to_numpy())
    axis_2 = pd.Series(df_02['Beta'].to_numpy())
    
    # Plot the dense graphs:
    dense_01 = axis_1.plot.kde() # Plot Patient
    dense_02 = axis_2.plot.kde() # Plot Control
    plt.legend(['Patient', 'Control'])
    plt.show()
    
def statistical_summary(data):
    
    data = data.drop(columns=['Participant Condition'])
    
    # Create data store dictionaries:
    min_dic = {}
    max_dic = {}
    mean_dic = {}
    
    # Initialise dictionaries:
    min_dic['Summary'] = ['Min']
    max_dic['Summary'] = ['Max']
    mean_dic['Summary'] = ['Mean']
    
    # Populate min dictionary:
    for c in data.columns:
        min_dic[c] = [data[c].min()]
        
    # Populate max dictionary:
    for c in data.columns:
        max_dic[c] = [data[c].max()]
        
    # Populate min dictionary:
    for c in data.columns:
        mean_dic[c] = [data[c].mean()]
        
    # Create dataframes:
    min_df = pd.DataFrame(min_dic)
    max_df = pd.DataFrame(max_dic)
    mean_df = pd.DataFrame(mean_dic)
    summary_df = pd.concat([min_df, max_df, mean_df])
    
    print(summary_df)
    
def neural_network(train, test):
    
    # We need to scale the features perhaps:
    
    # We need to create an array of class names:
    class_names = ['Patient', 'Control']
    
    print('Train')
    print(train)
    
    print('Test')
    print(test)
    
    train_drop = train.drop(['Image number', 'Bifurcation number', 'Artery (1)/ Vein (2)'])
    test_drop = test.drop(['Image number', 'Bifurcation number', 'Artery (1)/ Vein (2)'])
    
    train_x = train[['Alpha', 'Beta', 'Lambda1', 'Lambda2']]
    train_y = test['Participant Condition']
    
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[4]),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    
    
    
def main():
    
    # Get the dataset:
    df = pd.read_csv('../data/HIV.csv')
    patient_df = df.loc[df['Participant Condition'] == 'Patient'] # Get the patients as a df. 
    control_df = df.loc[df['Participant Condition'] == 'Control'] # Get the controls as a df.
    
    """ Data Visualisation """
    print('Min Max Mean')
    statistical_summary(data=df)
    
    # Present the graphs to the user:
    display_boxplot(data=df)
    display_density(df_01=patient_df, df_02=control_df)
    
    """ Machine Learning """
    
    # First step is to split the dataset into train and test df:
    train_df, test_df = train_test_split(df, test_size=0.1)
    
if __name__ == '__main__':
    main()