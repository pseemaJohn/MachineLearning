# Program Name : problem1.py
# Description : implement Perceptron Learning alogrithm for a linearly separable dataset\
# Input Parameter : <input-file>
# <input- file> : is a filename, containing 3 columns. Column1 - X-coordinates, Column2 - y-coordinates for each point
# Column3 - Label which has positive and negative value. \
# Output Parameter : create / write to a file called output1.csv, containing the 3 columns:
# Column1 and 2: contains values for weight and column3 is b.
# With each iteration of your PLA (each time we go through all examples), your program must print a new line to the output file,
# containing a comma-separated list of the weights w_1, w_2, and b in that order. Upon convergence, your program will stop,
# and the final values of w_1, w_2, and b will be printed to the output file (see example).
# This defines the decision boundary that your PLA has computed for the given dataset.S
# execute as : python3 problem1.py input1.csv output1.csv

import sys
import os
import numpy as np

def executePerceptronAlgo(X_feature_input,Y_label_input):
    weight = []
    devi = 1
    weight_curr = np.array([0,0,0])
    learn_rate = 0.1
    while True:
        w_prev = weight_curr
        for i,x1 in enumerate(X_feature_input):
            Y_new_label = np.dot(weight_curr,x1)
            if Y_new_label > 0:
                Y_new_label = 1
            else:
                Y_new_label = -1

            #delta_weight = learn_rate * (np.dot((Y_label_input[i] - Y_new_label),x1))
            #print("Value of delta is ",delta_weight)
            if Y_label_input[i] * Y_new_label <= 0:
                weight_curr = weight_curr + (Y_label_input[i] * x1)

        weight.append(weight_curr)
        if np.array_equal(w_prev,weight_curr) == True:
            break
    return weight

def processLearning():
    # check argument, if not 2 exit application
    if len(sys.argv) != 3:
        print(" Incorrect Argument provided \n python3 problem1.py <input-file> <output-file>")
        return 1
    X_feature_input = []
    Y_label_input = []

    fileN = sys.argv[1]
    # validate the arguments
    if os.path.exists(fileN) == False:
        print(" File doesn't exist")
        return 1

    with open(fileN, "r") as filePtr:
        for line in filePtr:
            outLine = list(map(int,(line.replace("\n","")).split(",")))
            Y_label_input.append(outLine[2])
            outLine.__delitem__(-1)
            outLine.append(1)
            X_feature_input.append(outLine)
    X_feature_input = np.array( X_feature_input)
    weight = executePerceptronAlgo( X_feature_input,Y_label_input)
    if weight != None:
        np.savetxt(sys.argv[2], weight, fmt="%d", delimiter=",")


    return 0

# Below line checks if the control of the program is main, __name__ variable provides the name of current process
if __name__ == '__main__':
    processLearning()
