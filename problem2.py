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


def executeGradientDescent(X_feature_input,Y_label_input):
    devi_limit = 0.001
    learn_rate_alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,0.11]
    #scale the data X
    mean_val = np.mean(X_feature_input,axis = 0)
    std_dev_val = np.std(X_feature_input,axis = 0)
    X_scaled_val = (X_feature_input - mean_val) / std_dev_val

    #insert the intercept
    X_scaled_val = np.insert(X_scaled_val,0,1,axis=1)
    final_output = []
    no_row,_ = X_scaled_val.shape
    Y_label_input = np.reshape(Y_label_input,(no_row,1))
    #beta_val = np.zeros([3,1])
    beta_val = np.array([0,0,0])
    for rate in learn_rate_alpha:
        iteration = 0
        beta_val = np.array([0,0,0])
        #print("****************Learning Rate = ",rate," ***************")
        while iteration < 100:
            beta_val_prev = beta_val
            risk_fac_temp = 0
            grad_desc_temp = 0
            for i,x1 in enumerate(X_scaled_val):
                Y_new_label = np.dot(x1,beta_val)
                temp_inp = Y_new_label - Y_label_input[i]
                risk_fac_temp = risk_fac_temp + ((temp_inp) ** 2)
                grad_desc_temp = grad_desc_temp + (temp_inp)*x1
                #print("Single difference ",input_fac)
            risk_fact = risk_fac_temp /(2 * no_row)
            beta_val = beta_val - rate * (grad_desc_temp / no_row)
            iteration += 1
        #print('\t alpha', rate, 'iterations',iteration, 'beta_intercept',float(beta_val[0]), 'beta_age',float(beta_val[1]), 'beta_weight',float(beta_val[2]))
        final_output.append([rate,iteration,beta_val[0],beta_val[1],beta_val[2]])

    #print(final_output)
    return final_output

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
            outLine = list(map(float,(line.replace("\n","")).split(",")))
            X_feature_input.append(outLine)
    X_feature_input = np.array( X_feature_input)
    Y_label_input = X_feature_input[:,-1]
    X_feature_input = np.delete(X_feature_input,-1,axis=1)
    final_list = executeGradientDescent( X_feature_input,Y_label_input)
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
            #with open(sys.argv[2], "wb") as f:
                #writer = csv.writer(f,delimiter=',',  quoting=csv.QUOTE_MINIMAL)
                #writer.writerows(final_list)


    return 0

# Below line checks if the control of the program is main, __name__ variable provides the name of current process
if __name__ == '__main__':
    processLearning()
