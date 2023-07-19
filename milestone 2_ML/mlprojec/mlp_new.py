from samples import producedata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
# loading datasets

# handwritten numerical face
ocr_training_samples = 5000
ocr_test_samples = 1000
digits_train_data, digits_train_labels = producedata(ocr_training_samples, "data/digitdata/trainingimages","data/digitdata/traininglabels") #return features and labels
digits_test_data, digits_test_labels  = producedata(ocr_test_samples, "data/digitdata/testimages", "data/digitdata/testlabels")

# edge image: classify as face or no face
face_training_samples=451
face_test_samples=150
face_train_data, face_train_labels = producedata(face_training_samples,"data/facedata/facedatatrain", "data/facedata/facedatatrainlabels",60)      #loading training data
face_test_data, face_test_labels = producedata(face_test_samples,"data/facedata/facedatatest", "data/facedata/facedatatestlabels",60)




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
    plt.xticks(fontsize=8)
    plt.xlabel("Parameters", fontsize=15)
    plt.ylabel("Score", fontsize= 15)   
    plt.show()
    




#mlp classifier

def classify_mlp(train_data, test_data, train_labels, test_labels):    
    
    # changing hyperparameters
    # first, we get the best hyperparamters using exhaustive search
    warnings.filterwarnings("ignore")
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    param_grid = {
    'hidden_layer_sizes': [5,10,15,(5,5),(5,10)], #'hidden_layer_sizes': [5], #
    'activation': ['identity','logistic','tanh','relu'],
    'solver': ['lbfgs','sgd','adam'] }
    #'max_iter': [1000],
    #'random_state': [5] }
    
    gridSearch = GridSearchCV(MLPClassifier(), param_grid, cv=cv,
                          refit=True,verbose=2, return_train_score=True,
                          n_jobs=-1)
    gridSearch.fit(train_data, train_labels)
    print("cv_results_: ", gridSearch.cv_results_)
    print('Score: ', gridSearch.best_score_)
    print('Parameters: ', gridSearch.best_params_)
    # then, we plot the parameter vs the score (uncomment the next line)
    plot_grid_search(gridSearch.cv_results_)


    # train, test accuracy
    h = gridSearch.best_params_['hidden_layer_sizes']
    a = gridSearch.best_params_['activation']
    s = gridSearch.best_params_['solver']
    mlp = MLPClassifier(hidden_layer_sizes=h,
                        activation=a,
                        solver=s,
                        random_state=5, max_iter=1000) 
    mlp.fit(train_data, train_labels)
    #print(mlp.score(train_data, train_labels)) = 1.0 when converge

    predictions_test = mlp.predict(test_data)
    test_score = accuracy_score(predictions_test, test_labels) # accuracy of predictions to actual labels
    print("\naccuracy score on test data: ", test_score)  
    rand = random.randint(0, 100) #output any random row to see 
    print("example of predicted output: ", predictions_test[:rand]) # predicted digits or faces OUTPUT
    print("whereas actual output: ", test_labels[:rand]) #actual label


    
classify_mlp(digits_train_data, digits_test_data, digits_train_labels, digits_test_labels)
classify_mlp(face_train_data, face_test_data, face_train_labels, face_test_labels)