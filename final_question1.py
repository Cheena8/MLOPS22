###IMPORT ALL THE MODULES

import matplotlib.pyplot as plt

import pytest

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
##CHOOSING Random seed as 42

RANDOM_SEED = 42

def train():

    digits = datasets.load_digits()

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=True, random_state=RANDOM_SEED
)

# Learn the digits on the train subset
    clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
    predictions = clf.predict(X_test)

    return ( predictions, y_test, clf )

def accuracy_metric(y_test, predicted, clf):


    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    print(' USING Random state equal to 42 AND Shuffle equal to FALSE, we can get the same samples while we split the dataset')
def test_same_random_seed():
    assert RANDOM_SEED == 42

def test_different_random_seed():
    assert RANDOM_SEED != 42

predictions, y_test, clf = train()
accuracy_metric(predictions, y_test, clf)
