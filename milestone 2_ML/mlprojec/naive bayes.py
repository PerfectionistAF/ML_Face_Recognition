

from samples import *
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt

training_samples = 5000
tst_samples = 1000
features, targets = producedata(training_samples)

#------------------------                   --------------------------------------      ----------------------------







tst_features_path = "data/digitdata/testimages"
tst_target_path =  "data/digitdata/testlabels"

tst_features, tst_targets  = producedata(tst_samples,tst_features_path,tst_target_path)

# training the model on training set
def all_work(var=0.01):

    x_points= []
    y_points = []
    norm = 1/var
    for i in range(int(var*norm),30,1):


        gnb = GaussianNB(var_smoothing=(i/norm))
        gnb.fit(features, targets)
        y_pred = gnb.predict(tst_features)

        x_points.append(i/norm)
        y_points.append(metrics.accuracy_score(tst_targets, y_pred) * 100)
        print("var = ",i/norm,"\tGaussian Naive Bayes model accuracy(in %):", y_points[i-1])
    print(x_points,"\n",y_points)



    plt.title("naive bayes", fontsize=20)
    plt.xlabel('sigma', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)

    plt.plot(x_points,y_points,marker='o')

    plt.show()
# making predictions on the testing set

all_work()


gnb = GaussianNB(var_smoothing=(0.05))
gnb.fit(features, targets)
y_pred = gnb.predict(tst_features)
print("var = ",0.05,"\tGaussian Naive Bayes model accuracy(in %):",(metrics.accuracy_score(tst_targets, y_pred) * 100))

# comparing actual response values (y_test) with predicted response values (y_pred)
face_training_samples=451
face_tst_samples=150

features,targets = producedata(face_training_samples,"data/facedata/facedatatrain","data/facedata/facedatatrainlabels",60)      #loading training data

tst_features,tst_targets = producedata(face_tst_samples,"data/facedata/facedatatest","data/facedata/facedatatestlabels",60)
print("\n-----------------------------------")
all_work(0.5)

gnb = GaussianNB(var_smoothing=(7))
gnb.fit(features, targets)
y_pred = gnb.predict(tst_features)
print("var = ",7,"\tGaussian Naive Bayes model accuracy(in %):",(metrics.accuracy_score(tst_targets, y_pred) * 100))