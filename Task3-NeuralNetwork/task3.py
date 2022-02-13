from cgi import test
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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
    
def data_prep(df):
    print('prep')
    raw_labels = df['Participant Condition'].to_numpy()
    
    le = preprocessing.LabelEncoder()
    le.fit(raw_labels)
    labels = le.transform(raw_labels)
    
    df['Labels'] = labels
    
    return df


def neural_network_10Fold(dataset, neurons):
    """
    We would like to uderstand which set of parameters are preferable,
    
    Employ 10-fold CV
    Create three ANN classifiers with 50, 500, 1000 neurons
    
    
    """
    
    dataset = data_prep(df=dataset)
    
    # Create the neural network:
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[5]),
        keras.layers.Dense(neurons, activation='sigmoid'),
        keras.layers.Dense(neurons, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Complie the model:
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    
    k = 10
    kf = KFold(n_splits=k, random_state=None)
    
    acc_score = []
    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    
    folds = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    acc_score = np.zeros(shape=(len(folds), 0))
    
    for train_index, test_index in kf.split(dataset):
        train = dataset.iloc[train_index]
        test = dataset.iloc[test_index]
        
        # Select the values to create X:
        train_x = train[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
        test_x = test[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
        
        # Select the values to create Y:
        train_y = train[['Labels']].to_numpy()
        test_y = test[['Labels']].to_numpy()
        
        # Reshape the Y values:
        train_y = np.reshape(train_y, (-1, len(train_y)))
        train_y = train_y[0]
        
        # Train the model here:
        history = model.fit(train_x, train_y, epochs=30)
        
        y_pred = model.predict_classes(test_x)
        acc = accuracy_score(y_pred, test_y)
        acc_score.append(acc)
        
    print(acc_score)
    print(folds)
    
    average_acc_score = sum(acc_score)/k
    
    return average_acc_score, acc_score

    
    
    
def neural_networkV2(train, test):
    # Select values to create X and Y:
    train_x = train[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    test_x = test[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    
    train_y = train[['Labels']].to_numpy()
    test_y = train[['Labels']].to_numpy()
    
    # Reshape Y values:
    train_y = np.reshape(train_y, (-1, len(train_y)))
    train_y = train_y[0]
    
    test_y = np.reshape(test_y, (-1, len(test_y)))
    test_y = test_y[0]
    
    print('X')
    print(train_x)
    
    print('Y')
    print(train_y)
    
    # Create neural network model: (Need a sigmoid function for hidden layers and a logistic function for the ouput)
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[5]),
        keras.layers.Dense(500, activation='sigmoid'),
        keras.layers.Dense(500, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # IGNORE THIS (THIS IS MY EXPERIENT)
    # opt = keras.optimizers.SGD(learning_rate=0.01)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    
    # Train the model:
    history = model.fit(train_x, train_y, epochs=60)
    
    # Present any metrics that it produces:
    history_df = pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    
def NeuralNetworkWithValidation(train, test, validation):
    
    # Create feature matrix's
    train_x = train[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    test_x = test[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    validation_x = validation[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    
    # Create label array:
    train_y = train[['Labels']].to_numpy()
    test_y = test[['Labels']].to_numpy()
    validation_y = validation[['Labels']].to_numpy()
    
    # Create neural network model: (Need a sigmoid function for hidden layers and a logistic function for the ouput)
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[5]),
        keras.layers.Dense(500, activation='sigmoid'),
        keras.layers.Dense(500, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    
    # Train the model:
    history = model.fit(train_x, train_y, epochs=60, validation_data=(validation_x, validation_y))
    
    # Present any metrics that it produces:
    history_df = pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    
    
def descision_tree(train, test):
    # Create a random forest classifier with 1000 trees.
    # The minimum number of samples required to be at a leaf node is either 5, 10
        # Show their accuracy on the test set.
        
    # N_estimators is the number of trees in the random forest:
    # 
    
    
def main():
    
    # Get the dataset:
    df = pd.read_csv('../data/HIV.csv')
    print(df)
    # patient_df = df.loc[df['Participant Condition'] == 'Patient'] # Get the patients as a df. 
    # control_df = df.loc[df['Participant Condition'] == 'Control'] # Get the controls as a df.
    
    # """ Data Visualisation """
    # print('Min Max Mean')
    # statistical_summary(data=df)
    
    # # Present the graphs to the user:
    # display_boxplot(data=df)
    # display_density(df_01=patient_df, df_02=control_df)
    
    
    
    """ 10 CV """
    average_acuracy, acc_scores = neural_network_10Fold(dataset=df, neurons=500)
    average_acuracy, acc_scores = neural_network_10Fold(dataset=df, neurons=50)
    average_acuracy, acc_scores = neural_network_10Fold(dataset=df, neurons=1000)
    
    
    
    
    
    
    
    return 'hello'
    
    
    """ Machine Learning """
    
    # prepped_df = data_prep(df=df)
    
    # # First step is to split the dataset into train and test df:
    # # train_df, test_df = train_test_split(prepped_df, test_size=0.1)
    
    # # Create the test set:
    # train_val_df, test_df = train_test_split(prepped_df, test_size=0.1)
    
    # # Create the validationand train set:
    # train_df, validation_df = train_test_split(train_val_df, test_size=0.1)
    
    # # neural_networkV2(train=train_df, test=test_df)
    # neural_networkV2(train=train_df, test=test_df)
    
    
if __name__ == '__main__':
    main()