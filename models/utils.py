import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

plt.rcParams["font.family"] = 'DejaVu Sans'

ACTIVITIES = {
    "chodz": 0, 
    "chodz_reka": 1,
    "wchod": 2, 
    "schod": 3, 
    "siada": 4,
    "upade": 5
}

labels = ["chodz", "chodz_reka", "wchod", "schod", "siada", "upade"]

def numbers_to_labels(numbers):
    result = pd.DataFrame(columns=["activity"])
    result["activity"] = [labels[number] for number in numbers]
    return result

def plot_confusion_matrix(classifier, X_test, y_test, classes, normalize='true', cmap=plt.cm.Blues):
    
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=classes,
        cmap=cmap,
        normalize=normalize,
    )

    if normalize:
        disp.ax_.set_title("Normalized confusion matrix")
        print("Normalized confusion matrix")
    else:
        disp.ax_.set_title("onfusion matrix, without normalization")
        print("Confusion matrix, without normalization")

    print(disp.confusion_matrix)
    plt.show()   