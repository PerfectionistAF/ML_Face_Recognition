from samples import producedata
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
#from hypopt import GridSearch
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
# loading datasets

# handwritten numerical face
ocr_training_samples = 5000
ocr_test_samples = 1000
n_validation = 1000
digits_train_data, digits_train_labels = producedata(ocr_training_samples, "data/digitdata/trainingimages","data/digitdata/traininglabels") #return features and labels
digits_test_data, digits_test_labels  = producedata(ocr_test_samples, "data/digitdata/testimages", "data/digitdata/testlabels")
#digits_val_data, digits_val_labels = producedata(n_validation, "data/digitdata/validationimages", "data/digitdata/validationlabels")

# edge image: classify as face or no face
face_training_samples=451
face_test_samples=150
face_train_data, face_train_labels = producedata(face_training_samples,"data/facedata/facedatatrain", "data/facedata/facedatatrainlabels",60)      #loading training data
face_test_data, face_test_labels = producedata(face_test_samples,"data/facedata/facedatatest", "data/facedata/facedatatestlabels",60)
#TODO face_val_data, face_val_labels = producedata(50, "data/facedata/facedatavalidation", "data/facedata/facedatavalidationlabels")




#decision tree classifier

def classify_tree(train_data, test_data, train_labels, test_labels): #val_data, val_labels):    
    
    
    # changing hyperparameters
    # first, we get the best hyperparamters using exhaustive search
    warnings.filterwarnings("ignore")
    #for no cross-validation, cv=cv
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    #for cross-validation with digits_val_data, use cv=pds
    # Create a list where train data indices are -1 and validation data indices are 0
    #val_and_train = train_data + val_data
    #split_index = [-1 if x in X_train.index else 0 for x in val_and_train.index]
    # Use the list to create PredefinedSplit
    #pds = PredefinedSplit(test_fold = split_index) and then set cv=pds and fit with all data
    param_grid = {
    "max_depth":[None, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40, 100, 500, 1000] }
    tree = DecisionTreeClassifier()
    gridSearch = GridSearchCV(tree, param_grid, cv=cv, # splits 
                              refit = True, verbose=3, return_train_score=True, 
                              n_jobs=-1)
    gridSearch.fit(train_data, train_labels) #all data
    #print("cv_results_: ", gridSearch.cv_results_)

    print('Score: ', gridSearch.best_score_)
    print('Parameters: ', gridSearch.best_params_)
    
    
   # then, we plot the parameter vs the score
    y_values = gridSearch.cv_results_['mean_test_score']
    x_values = [-1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40, 100, 500, 1000]
    
    print ('plot x: ', x_values )
    print ('plot y: ', y_values )
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    # Plot Grid search scores
    plt.plot()
    plt.plot(x_values, y_values, marker='o')
    plt.xscale('log')
    plt.xlabel("Max Tree Depth", fontsize=15)
    plt.ylabel("Score", fontsize= 15)   
    plt.show()

    # train, test accuracy
    m = gridSearch.best_params_["max_depth"]
    tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=m)
    decision_tree = tree.fit(train_data,train_labels)
    prediction = tree.predict(test_data)
    print("\nprediction accuracy on test data: ",tree.score(test_data,test_labels)*100,"%")

    rand = random.randint(0, 100) #output any random row to see 
    print("example of predicted output: ", prediction[:rand]) # predicted digits or faces OUTPUT
    print("whereas actual output: ", test_labels[:rand]) #actual label
    r = export_text(decision_tree)
    #print("\n", r, "\n")


    
classify_tree(digits_train_data, digits_test_data, digits_train_labels, digits_test_labels)
             #digits_val_data, digits_val_labels)
classify_tree(face_train_data, face_test_data, face_train_labels, face_test_labels)
             #face_val_data, face_val_labels)