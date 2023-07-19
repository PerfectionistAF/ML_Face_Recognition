from samples import producedata
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit
# loading datasets

# handwritten numerical face
ocr_training_samples = 5000
ocr_test_samples = 1000
n_validation = 1000
digits_train_data, digits_train_labels = producedata(ocr_training_samples, "data/digitdata/trainingimages","data/digitdata/traininglabels") #return features and labels
digits_test_data, digits_test_labels  = producedata(ocr_test_samples, "data/digitdata/testimages", "data/digitdata/testlabels")
digits_val_data, digits_val_labels = producedata(n_validation, "data/digitdata/validationimages", "data/digitdata/validationlabels")


# edge image: classify as face or no face
face_training_samples=451
face_test_samples=150
face_val_samples=200
face_train_data, face_train_labels = producedata(face_training_samples,"data/facedata/facedatatrain", "data/facedata/facedatatrainlabels",60)      #loading training data
face_test_data, face_test_labels = producedata(face_test_samples,"data/facedata/facedatatest", "data/facedata/facedatatestlabels",60)
face_val_data, face_val_labels = producedata(face_val_samples, "data/facedata/facedatavalidation", "data/facedata/facedatavalidationlabels", 60)
#print("lucy in the sky: ", face_val_data)
#print("with diamonds: ", face_val_labels)


# visualizes results of hyperparameter exhaustive search

def plot_grid_search(cv_results):
    
    # get five random parameters combintations (x values) and their mean_score (y_values)
    x_values = []
    y_values = []
    for i in range(5):#len(cv_results)):
        rand = random.randint(0, len(cv_results['params']) - 1)
        d = cv_results['params'][rand] # {'activation': 'tanh', 'hidden_layer_sizes': 5, 'solver': 'lbfgs'}
        value = ", ".join(map(str, d.values())) #tanh, 5, lbfgs
        x_values.append(value)
        y_values.append(cv_results['mean_test_score'][rand])
    #data = dict(zip(x_values,y_values)) # [{'tanh, 5, lbfgs': 0.7}, {}, {}, etc]
    print("x_values: ", x_values)
    print("y_values: ", y_values)
    
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    # Plot Grid search scores
    
    plt.bar(x_values, y_values, width=0.5)
    plt.xticks(fontsize=8, rotation=90)
    plt.xlabel("Parameters", fontsize=15)
    plt.ylabel("Score", fontsize= 15)   
    plt.show()
    




#svm classifier

def classify_svm(train_data, test_data, train_labels, test_labels, val_data, val_labels):    
    
    # changing hyperparameters
    # first, we get the best hyperparamters using exhaustive search
    warnings.filterwarnings("ignore")
    #for no cross-validation, cv=cv
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1) 
    #for cross-validation with digits_val_data, use cv=pds
    # Create a list where train data indices are -1 and validation data indices are 0
    val_and_train = np.concatenate([face_train_data, face_val_data])
    #split_index = [-1 if x in train_data.index else 0 for x in val_and_train.index]
    # Use the list to create PredefinedSplit
    #pds = PredefinedSplit(test_fold = split_index) #and then set cv=pds and fit with all data
    param_grid = {
    "C":[0.1, 1, 10, 100],
    "kernel":['rbf','linear','poly','sigmoid'],
    "gamma":[0.1,'auto','scale'] }
    svc = svm.SVC()
    gridSearch = GridSearchCV(svc, param_grid, cv=cv, #splits 
                              refit = True, verbose=3, return_train_score=True, 
                              n_jobs=-1)
    gridSearch.fit(train_data, train_labels) #all data
    #print("cv_results_: ", gridSearch.cv_results_)

    print('Score: ', gridSearch.best_score_)
    print('Parameters: ', gridSearch.best_params_)
    # then, we plot the parameter vs the score (uncomment the next line)
    plot_grid_search(gridSearch.cv_results_)


    # train, test accuracy
    c = gridSearch.best_params_['C']
    k = gridSearch.best_params_['kernel']
    g = gridSearch.best_params_['gamma']
    svc = SVC(C=c, kernel=k, gamma=g,
              random_state=1)
    fit = svc.fit(train_data, train_labels)
    labels_pred = fit.predict(test_data)
    #print("svc score: ", svc.score(test_data, test_labels))
    print("\naccuracy score on test data: ", accuracy_score(test_labels, labels_pred))

    rand = random.randint(0, 100) #output any random row to see 
    print("example of predicted output: ", labels_pred[:rand]) # predicted digits or faces OUTPUT
    print("whereas actual output: ", test_labels[:rand], "\n") #actual label


    
classify_svm(digits_train_data, digits_test_data, digits_train_labels, digits_test_labels,
             digits_val_data, digits_val_labels)
classify_svm(face_train_data, face_test_data, face_train_labels, face_test_labels,
             face_val_data, face_val_labels)