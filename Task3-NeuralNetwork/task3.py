from cgi import test
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
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
    std_dic = {}
    
    # Initialise dictionaries:
    min_dic['Summary'] = ['Min']
    max_dic['Summary'] = ['Max']
    mean_dic['Summary'] = ['Mean']
    std_dic['Summary'] = ['Standard Deviation']
    
    # Populate min dictionary:
    for c in data.columns:
        min_dic[c] = [data[c].min()]
        
    # Populate max dictionary:
    for c in data.columns:
        max_dic[c] = [data[c].max()]
        
    # Populate min dictionary:
    for c in data.columns:
        mean_dic[c] = [data[c].mean()]
        
    for c in data.columns:
        std_dic[c] = [data[c].std()]
        
    # Create dataframes:
    min_df = pd.DataFrame(min_dic)
    max_df = pd.DataFrame(max_dic)
    mean_df = pd.DataFrame(mean_dic)
    std_df = pd.DataFrame(std_dic)
    summary_df = pd.concat([min_df, max_df, mean_df, std_df])
    
    print(summary_df)
    
def data_visutalisation(data):
    # Get the patient and control labels as separate dataframes:
    patient_df = data.loc[data['Participant Condition'] == 'Patient'] # Get the patients as a df. 
    control_df = data.loc[data['Participant Condition'] == 'Control'] # Get the controls as a df.
    
    # Min Max Mean
    print('Min Max Mean')
    statistical_summary(data=data)
    
    # Present the graphs to the user:
    display_boxplot(data=data)
    display_density(df_01=patient_df, df_02=control_df)
    
def data_prep(df):
    raw_labels = df['Participant Condition'].to_numpy() # Gather the labels as numpy array.
    le = preprocessing.LabelEncoder() # Instantiate a sklearn Label Encoder.
    le.fit(raw_labels) # Fit the raw labels (the strings) into binary.
    labels = le.transform(raw_labels) # Trasform into the labels variable.
    df['Labels'] = labels # Add the labels to the dataframe.
    return df

def neural_network_10Fold(dataset, neurons):
    
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
    
    return average_acc_score


def neural_network_classifier(train, test, epochs):
    # Select values to create X and Y:
    train_x = train[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    test_x = test[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    
    train_y = train[['Labels']].to_numpy()
    test_y = test[['Labels']].to_numpy()
    
    # Reshape Y values:
    train_y = np.reshape(train_y, (-1, len(train_y)))
    train_y = train_y[0]
    
    test_y = np.reshape(test_y, (-1, len(test_y)))
    test_y = test_y[0]
    
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
    history = model.fit(train_x, train_y, epochs=epochs)
    
    # Present any metrics that it produces:
    history_df = pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    
    # Calculate the accuracy of the Neural network model:
    model_predicts = model.predict_classes(test_x)
    
    # Retreive the accuracy score of the Neural network:
    acc_score = accuracy_score(test_y, model_predicts)
    
    return acc_score
    
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
    
    
def random_forest(train, test, min_leaf_samples):
    
    # Create the machine learning model:
    model = RandomForestClassifier(min_samples_leaf=min_leaf_samples, n_estimators=1000)
    
    # Select values to create X and Y:
    train_x = train[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    test_x = test[['Alpha', 'Beta', 'Lambda','Lambda1', 'Lambda2']].to_numpy()
    
    train_y = train[['Labels']].to_numpy()
    test_y = test[['Labels']].to_numpy()
    
    # Reshape Y values:
    train_y = np.reshape(train_y, (-1, len(train_y)))
    train_y = train_y[0]
    
    test_y = np.reshape(test_y, (-1, len(test_y)))
    test_y = test_y[0]
    
    # Fit the training data to the random forest:
    model.fit(train_x, train_y)
    
    model_predicts = model.predict(test_x)
    
    """ Check the accuracy metric"""
    acc = accuracy_score(test_y, model_predicts)
    
    return acc
    
    
def random_forest_10Fold(dataset, trees):
    
    # First prep the dataset to get the labels as either 1 or 0:
    dataset = data_prep(df=dataset)
    
    # Create the random forest model:
    model = RandomForestClassifier(min_samples_leaf=10, n_estimators=trees)
    
    # Initialise the variables for K Cross Validation:
    k = 10 # State how many splits/folds whill be carried out for the KFold.
    kf = KFold(n_splits=k, random_state=None)
    
    acc_scores = []
    
    # Commence the model training:
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
        
        test_y = np.reshape(test_y, (-1, len(test_y)))
        test_y = test_y[0]
        
        # Train the model:
        model.fit(train_x, train_y)
        
        # Create the predictions:
        model_predicts = model.predict(test_x)
        
        acc = accuracy_score(test_y, model_predicts)
        
        # Append the accuracy scores:
        acc_scores.append(acc)
    
    average_acc = sum(acc_scores) / k
    
    return average_acc
    
    
def main():
    
    """
    Task03 TODO:
    
    3.2: 
        - Split the dataset into 9/1 [X]
        - Train a neural network with 500 neurons [X]
        - Calculate the accuracy (the fraction of properly classified cases) using the test set. [X ]
        - Report the process of splitting the data, as well as report the steps undertaken to train the ANN in detail. []
        - Try a different number or epochs and monitor how accuracy changes as the algorithm keeps learning. []
            - Plot epochs in x and accuracy in the y -> It should show the change inaccuracy as the epochs increase. []
            
        - Train a random forest with 1000 trees[X]
        - Train it with a leaf node of either 5/10 -> Report the steps and accuracy  [X]
    
    3.3:
        - Train an ANN with 50, 500, 1000 neurons in each layer (two hidden layers) [X]
        - Train random forests with 50, 500, 10000 trees, minimum leaf node = 10 []
        
        - Use the 10 fold CV method to choose the best number of neurons or number of trees for ANN and random forest
            - Report the process involved when applying CV to each model. []
            - Report the mean accuracy results for each set of parameters []
            - Which parameters should we use for the two methods
        
        - Which method is the best ANN or Random Forest?
        - Please discuss and justify your choice, reflecting upon your knowledge thus far.
    """
    
    # Get the dataset:
    df = pd.read_csv('../data/HIV.csv')
    
    #data_visutalisation(data=df)
    
    """ Data preparation """
    #  Create the column for the labels:
    prepped_df = data_prep(df=df) # This is to turn the labels 'Patient' 'Control' into 1s and 0s.
    
    # First step is to split the dataset into train and test df:
    train_df, test_df = train_test_split(prepped_df, test_size=0.1) 
    
    """ Machine Learning without CV """
    neural_network_metric = neural_network_classifier(train_df, test_df, epochs=30)
    random_forest_accuracy_leaf5 = random_forest(train_df, test_df, min_leaf_samples=5)
    random_forest_accuracy_leaf10 = random_forest(train_df, test_df, min_leaf_samples=10)
    
    print('The accuracy for the neural netowrk is: ', neural_network_metric)
    print('The accuracy for the random forest with min_leaf_samples=5 is: ', random_forest_accuracy_leaf5)
    print('The accuracy for the random forest with min_leaf_samples=10 is: ', random_forest_accuracy_leaf10)
    
    
    """ 10 CV """
    nn_average_accuracy_50_neurons = neural_network_10Fold(dataset=df, neurons=50)
    nn_average_accuracy_500_neurons = neural_network_10Fold(dataset=df, neurons=500)
    nn_average_accuracy_1000_neurons = neural_network_10Fold(dataset=df, neurons=1000)
    
    
    
    print('Forest one')
    rf_average_accuracy_50_trees = random_forest_10Fold(dataset=df, trees=50)
    print('Forest two')
    rf_average_accuracy_500_trees = random_forest_10Fold(dataset=df, trees=500)
    print('Forest three')
    rf_average_accuracy_10000_trees = random_forest_10Fold(dataset=df, trees=10000)
    
    print('Training without 10 Fold CV Method')
    
    
    print('Training with the 10 Fold CV Method')
    print('The accuracy of the 10 CV Neural Network with 50 neurons is: ', nn_average_accuracy_50_neurons)
    print('The accuracy of the 10 CV Neural Network with 500 neurons is: ', nn_average_accuracy_500_neurons)
    print('The accuracy of the 10 CV Neural Network with 100 neurons is: ', nn_average_accuracy_1000_neurons)
    
    print('The accuracy of the 10 CV Random Forest with 50 trees is: ', rf_average_accuracy_50_trees)
    print('The accuracy of the 10 CV Random Forest with 500 trees is: ', rf_average_accuracy_500_trees)
    print('The accuracy of the 10 CV Random Forest with 10000 trees is: ', rf_average_accuracy_10000_trees)
    
    
if __name__ == '__main__':
    main()