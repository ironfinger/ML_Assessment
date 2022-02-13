

# Data Prep:
    - Train test split of 9:1

# Designing Algorithms:
    
## Neural Network:
    - Need to design a neural network for classifying patients as patient or control based on [].
    - Fully connected nn architecture with two hidden layers.
        - hidden layers [500 neurons] -> sigmoid activation
        - output layer [1 neuron] -> sigmoid activation
    - 

## Descision Tree:
    - 

## Tasks for the things:
    - Use 10 fold CV method to choose the best number of neurons or number of trees for ANN and random forests
        - Report the processes involved when applying CV.
        - Report the mean accuracy results for each set of params.
        - Which parameters should we use for each of the two methods.
    - Eval
        - Which method is the best
        - Discuss and justify your choice.

# 10 fold cross validation:

This method divides the dataset random into 10 parts.
    - 9 of which are used for training and the last is reserved for training.
    - We repeat this procedure 10 times each time reserving a different tenth.

# Medium article notes:

- Machine learning models often fail to generalize well on data it has not been trained on. Sometimes, it fails miserably. To be sure that the model can perform well on unseen data, we use a re-sampling technique called Cross-Validation.

- We follow a simple approach of splitting the data into 3 parts:
    - Train validataoin and test sets, but this technique does not work well for cases when we don't have a large dataset. 

    - QUOTE: 'When we have limited data, diving the dataset into Train and Validation sets may cause some datapoints with useful information to be excluded from the training procuedure, and thus the model fails to learn the data distribution properly.

- K-Fol CV (cross validation) gives a model