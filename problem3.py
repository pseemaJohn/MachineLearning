# Program Name : problem2.py
# Description : implement Linear Regression  with multiple features using gradient descent\
# Input Parameter : <input-file>
# <input- file> : is a filename, series of data points\
# Each point is a comma-separated ordered triple, representing age, weight, and height
# Output Parameter : create / write to a file called output1.csv, containing the 5 columns:
# containing a comma-separated list of alpha, number_of_iterations, b_0, b_age, and b_weight in that order.
# execute as : python3 problem2.py input2.csv output2.csv

import sys
import os
import csv
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

learn_name = {'svm_linear':{'estimator':svm.SVC(),'parameter':{'kernel':['linear'],'C':[0.1, 0.5, 1, 5, 10, 50, 100]}},
              'svm_polynomial':{'estimator':svm.SVC(),'parameter':{'kernel':['poly'],'C':[0.1, 1, 3],'gamma':[0.1, 0.5],'degree':[4, 5, 6]}},
              'svm_rbf':{'estimator':svm.SVC(),'parameter':{'kernel':['rbf'],'C':[0.1, 0.5, 1, 5, 10, 50, 100],'gamma': [0.1, 0.5, 1, 3, 6, 10]}},
              'logistic':{'estimator':LogisticRegression(),'parameter':{'C':[0.1, 0.5, 1, 5, 10, 50, 100]}},
              'knn':{'estimator':KNeighborsClassifier(),'parameter':{'n_neighbors': range(1, 50), 'leaf_size': range(5, 60, 5)}},
              'decision_tree':{'estimator':DecisionTreeClassifier(),'parameter':{'max_depth': range(1, 50), 'min_samples_split': range(2, 10)}},
              'random_forest':{'estimator':RandomForestClassifier(),'parameter':{'max_depth': range(1, 50), 'min_samples_split': range(2, 10)}}
              }

def executeSVMclassify(X_feature_input,Y_label_input):
    #x_train, x_test, y_train, y_test = train_test_split(X_feature_input,Y_label_input,test_size=0.40,train_size=0.60,random_state=42,stratify=Y_label_input)
    spilt_data = StratifiedShuffleSplit(5,test_size=0.40,train_size=0.60,random_state=42)
    for train_ind,test_ind in spilt_data.split(X_feature_input, Y_label_input):
        x_train, x_test = X_feature_input[train_ind],X_feature_input[test_ind]
        y_train, y_test = Y_label_input[train_ind],Y_label_input[test_ind]

    #print("X train ",x_train)
    #print("Y train ",y_train)
    #print("X test ",x_test)
    #print("Y test ",y_test)

    #estimator = svm.SVC()
    with open("output3.csv", "w") as file_write:
        for type in learn_name:
            print(learn_name[type]['parameter'])
            classifierObj = GridSearchCV(learn_name[type]['estimator'],learn_name[type]['parameter'] ,cv=5)
            classifierObj.fit(x_train,y_train)
            train_score = classifierObj.best_score_
            print(type,": Best param ",classifierObj.best_params_)
            test_score = classifierObj.score(x_test,y_test)
            #print("Test Score ", test_score)
            file_write.write(type +"," + str('{0:.2f}'.format(train_score)) + "," + str(test_score) + "\n")
            #print(strtowrite)
    return


def processLearning():
    # check argument, if not 2 exit application
    if len(sys.argv) != 3:
        print(" Incorrect Argument provided \n python3 problem1.py <input-file> <output-file>")
        return 1
    X_feature_input = []

    fileN = sys.argv[1]
    # validate the arguments
    if os.path.exists(fileN) == False:
        print(" File doesn't exist")
        return 1

    with open(fileN, "r") as filePtr:
        for line in filePtr:
            if line.__contains__("label"):
                continue
            outLine = list(map(float,(line.replace("\n","")).split(",")))
            X_feature_input.append(outLine)

    X_feature_input = np.array( X_feature_input)
    Y_label_input = X_feature_input[:,-1]
    X_feature_input = np.delete(X_feature_input,-1,axis=1)
    #print(X_feature_input)
    #print(Y_label_input)
    #for type in learn_name:
    final_list = executeSVMclassify( X_feature_input,Y_label_input)
    return
    if final_list != None:
        with open(sys.argv[2], "w") as file_write:
            for val_list in final_list:

                for i,item in enumerate(val_list):
                    if i == 4:
                        file_write.write(str(item))
                    else:
                        file_write.write(str(item) + ',')
                file_write.write('\n')
            #np.savetxt(sys.argv[2], final_list, delimiter=",")
    return 0

# Below line checks if the control of the program is main, __name__ variable provides the name of current process
if __name__ == '__main__':
    processLearning()
