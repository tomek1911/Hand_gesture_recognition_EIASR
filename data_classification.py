import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import os
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, metrics
from sklearn.metrics import plot_confusion_matrix

from feature_extraction import FeatureExtraction as fe

class DataClassification:
    """Class provides tools to classify images using features."""
    
    @staticmethod
    def getXyfromCSV(path_features_csv):
        X = []
        y = []

        with open(path_features_csv) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)
            for row in csv_reader:
                temp_row = row
                X_row = temp_row[:len(temp_row)-1]
                y_row = temp_row[len(temp_row)-1:]
                y_row = y_row[0]
                X.append(X_row)
                y.append(y_row)
        
        return X, y
    
    @staticmethod
    def generate_conf_matrix(y_pred, y_test, save=False, title='Confusion Matrix - HOG - SVM', matrix_name='Confusion_matrix_hog_svm.png'):

        my_dir = os.getcwd()
        plot_dir = os.path.join(my_dir, "Plots")

        matrix = confusion_matrix(y_pred, y_test)
        plt.matshow(matrix)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.show()        
        
        if save:
            plt.savefig(plot_dir + "//confMatrix_hog_svm.png")

    @staticmethod
    def confusion_matrix2(clf, X, y, printInTerminal = False):

        my_dir = os.getcwd()
        plot_dir = os.path.join(my_dir, "Plots")

        np.set_printoptions(precision=2)
        title = 'Confusion Matrix - HOG - SVM'
        lst = list(set(y))
        classNames = sorted(lst)

        disp = plot_confusion_matrix(clf, X, y, display_labels=classNames, cmap=plt.cm.Blues, normalize=None)
        disp.ax_.set_title(title)
        if printInTerminal == True:
            print(title)
            print(disp.confusion_matrix)            

        plt.savefig(plot_dir + "//confMatrix_hog_svm.png")

    @staticmethod
    def fitKnn(X,y, test_split_ratio=0.25, print_res = False):

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=test_split_ratio)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        knn = KNeighborsClassifier() #TODO grid-search, cross-validation parameters-tuning
        knn.fit(X_train, y_train)

        if print_res==True:
                print(f"Training dataset accuracy: {knn.score(X_train, y_train):.3f}, test dataset accuracy: {knn.score(X_test, y_test):.3f}.")

        return knn
    
    @staticmethod
    def fitSVM(X,y,test_split_ratio=0.2, print_res=False, print_detailed_res=False, confusionMatrix = False):
  
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True, test_size=test_split_ratio)

        # fe.project2D_PCA(X,y)
     
        # print(f"Training set size: {len(X_train)}x{len(X_train[0])}")
        # X_train, X_test = fe.applyPCA(X_train, X_test)
        # print(f"Training set size after PCA: {len(X_train)}x{len(X_train[0])}")
                
        param_grid = [
            {'C': [1], 'kernel': ['linear']},
            # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]
        
        svc = svm.SVC()
        clf = GridSearchCV(svc, param_grid, verbose=1)
        clf.fit(X_train, y_train)
        filename = 'SVM_hogRGB_noPCA.sav'
        joblib.dump(clf, filename)

        y_pred = clf.predict(X_test)
        
        if print_res == True:
            print(f"Training dataset accuracy: {clf.best_estimator_.score(X_train, y_train):.3f}, test dataset accuracy: {clf.best_estimator_.score(X_test, y_test):.3f}.")
                   
        if print_detailed_res == True:
            print("Classification report for - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred)))
        
        if confusionMatrix == True:
            # DataClassification.generate_conf_matrix(y_pred, y_test, save = True)
            DataClassification.confusion_matrix2(clf, X_test, y_test)

        return clf

