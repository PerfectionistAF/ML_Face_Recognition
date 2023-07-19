from samples import *
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from importlib import reload

plt = reload(plt)

training_samples = 5000
tst_samples = 1000
features, targets = producedata(training_samples)

tst_features_path = "data/digitdata/testimages"
tst_target_path =  "data/digitdata/testlabels"


tst_features, tst_targets  = producedata(tst_samples,tst_features_path,tst_target_path)


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
#training_features, test_features, training_targets, test_targets = train_test_split(features, targets,
                                                                                   # test_size=0.3)  # 70% training and 30% test


# train test sets(lstoflsts[[]], lst of the same size)

# building KNN classifier

from sklearn import metrics

def train_knn_classifier(features, target, k=5):
    model = KNeighborsClassifier(n_neighbors=k)  # choosing the hyper parameter K = xxx
    # print(k)

    # Train the model using the training sets
    model.fit(features, target)  # feeding data to the model
    return model


def predict_output(model, tst_data=[[0, 2]]):
    # Predict Output
    predicted = model.predict(tst_data)

    return predicted













# EVALUATION

# Import scikit-learn metrics module for accuracy calculation




def all_the_work(i=3,title=None,xlbl=None,ylbl=None):

    x_points = []
    y_points = []

    while i <= (training_samples ** 0.5):                                    # no of k neighbours should be odd

        trained_model = train_knn_classifier(features, targets, i)          # train model and predict output
        predicted_targets = predict_output(trained_model, tst_features)     #predict the output
        acc = metrics.accuracy_score(tst_targets, predicted_targets)        # Model Accuracy, how often is the classifier correct?
        #visualization
        x_points.append(i)
        y_points.append(acc)
        print("k :",i,"\tAccuracy:",acc)
        i+=2

    print(list(zip(x_points,y_points)))
    plt.plot(x_points, y_points,marker='o')
    plt.title(title,fontsize = 20)
    plt.xlabel(xlbl,fontsize = 20)
    plt.ylabel(ylbl,fontsize = 20)
    plt.show()
all_the_work(title='knn',xlbl='k-neighbours',ylbl='accuracy')
"""KNN DOESNOT PERFORM PROPERLY WITH LARGE NUMBER OF FEATURES"""
trained_model = train_knn_classifier(features, targets, 7)          # train model and predict output
predicted_targets = predict_output(trained_model, tst_features)
acc = metrics.accuracy_score(tst_targets, predicted_targets)        # Model Accuracy, how often is the classifier correct?
print("k :",7,"\tAccuracy:",acc)



                    # """------------------------------classifying face images---------------------------"""

features,targets = producedata(451,"data/facedata/facedatatrain","data/facedata/facedatatrainlabels",60)      #loading training data

tst_features,tst_targets = producedata(150,"data/facedata/facedatatest","data/facedata/facedatatestlabels",60)
print("\n-----------------------------------")
all_the_work(title='KNN',xlbl='K-neighbours',ylbl='accuracy')

trained_model = train_knn_classifier(features, targets, 55)          # train model and predict output
predicted_targets = predict_output(trained_model, tst_features)
acc = metrics.accuracy_score(tst_targets, predicted_targets)        # Model Accuracy, how often is the classifier correct?
print("k :",55,"\tAccuracy:",acc)


