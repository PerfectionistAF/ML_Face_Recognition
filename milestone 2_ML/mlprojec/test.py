from samples import producedata
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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
print("lucy in the sky: ", face_val_data, face_val_data.shape)
print("with diamonds: ", face_train_data, face_train_data.shape)

val_and_train = np.concatenate([face_train_data, face_val_data])
print("val and train:", val_and_train)
print(val_and_train.shape)