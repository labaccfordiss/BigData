

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# summarize model performance
def summarize_performance(scores,item):
    sensors = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]
    pos = np.arange(len(sensors))
    plt.plot(pos, scores, color='blue')
    #plt.bar(pos, scores, color='blue', edgecolor="black")
    plt.xticks(pos, sensors)
    plt.xlabel('Sensors', fontsize=16)
    plt.ylabel(item, fontsize=16)
    plt.title('Ranking_' + item , fontsize=20)


