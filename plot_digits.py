# Standard scientific Python imports
import matplotlib.pyplot as plt
import pandas as pd
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics,tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from joblib import dump, load
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {}
params["gamma"] = gamma_list
params["C"] = c_list

h_param_comb = [{"gamma": g, "C": c} for g in params['gamma'] for c in params['C']]
digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
 n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
##################clf = svm.SVC(gamma=0.001)
##################clf2 = tree.DecisionTreeClassifier()

# Split data into 50% train and 50% test subsets
#################X_train, X_test, y_train, y_test = train_test_split(
################# data, digits.target, test_size=0.5, shuffle=False
#################)
##CODE TO SPLIT THE DATA IN 5 SPLITS
## USING KFOLD for the same
df=pd.DataFrame(columns=['run','svm','decisiontree'])
kfold = KFold(5)
X=data
y=digits.target
idxnum=1
for train_index, test_index in kfold.split(data):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    pred_test = clf.predict(X_test)
    AccuracyofDectree=metrics.accuracy_score(pred_test, y_test )

    clf2 = svm.SVC(gamma = 0.001)
    clf2.fit(X_train,y_train)
    predic_test= clf2.predict(X_test)
    AccuracySVM=metrics.accuracy_score(predic_test, y_test)
    print("accuracy of SVM")
    print(AccuracySVM)
    print("accuracy of Decision tree")
    print(AccuracyofDectree)
    idxnum+=1
    df.at[idxnum,'run']=idxnum
    df.at[idxnum,'svm']=AccuracySVM
     df.at[idxnum,'decisiontree']=AccuracyofDectree

    print(df.head())
    #actual_model_path = tune_and_save(clf2,X_train, y_train, X_test, y_test, AccuracySVM,h_param_comb, model_path= None)
    model_path="svm"+".joblib"
    dump(clf2,model_path)

#kf = KFold(n_splits=5)
#kf.get_n_splits(data)
#KFold(n_splits=5, random_state=None, shuffle=False)
#for train_index, test_index in kf.split(data):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = data[train_index], data[test_index]
#    y_train, y_test = data[train_index], data[test_index]

# Learn the digits on the train subset
clf.fit(X_train, y_train)
## FIRST ONE SVC AND SECOND ONE DECISION TREE
clf2  = clf2.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
predicted2 = clf2.predict(X_test)

#best_model = load(actual_model_path)
#predic = best_model.predict(X_test)
#dump(best_model, f'models/{best_model}')
#pred_image_viz(X_test, predic)
###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
    print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
print(f"Classification report for classifier{clf2}:\n" f"{metrics.classification_report(y_test, predicted2)}\n")

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
