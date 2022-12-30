

import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

def conf_mat(sname,y_test, y_pred, label):
    mat = confusion_matrix(y_test, y_pred)
    label.classes_=['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'walking']
    plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(6, 6))
    plt.title("Confusion matrix " + sname)
    print(mat)





