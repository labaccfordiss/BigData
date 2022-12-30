

import pickle
import os
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
def plot_learningCurve(histories, sname):
    plt.suptitle('Learning curve ' + sname )
    # plot loss
    plt.subplot(121)
    plt.title('Cross Entropy Loss')
    plt.plot(histories['loss'], color='blue', label='train')
    # plot accuracy
    plt.subplot(122)
    plt.title('Classification Accuracy')
    plt.plot(histories['accuracy'], color='orange', label='train')


